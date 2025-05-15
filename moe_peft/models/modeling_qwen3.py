from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.qwen3 import modeling_qwen3
from transformers.utils import is_flash_attn_2_available

from moe_peft.common import (
    FeedForward,
    LLMCache,
    LLMModelInput,
    flash_attention_forward,
)
from moe_peft.executors import executor
from moe_peft.models.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaEmbedding,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)
from moe_peft.utils import copy_parameters


@dataclass
class Qwen3Config(LlamaConfig):
    use_sliding_window_: bool = False
    max_window_layers_: int = None
    sliding_window_: int = None
    attention_bias_: bool = False
    attention_dropout_: float = 0.0


class Qwen3FlashAttention(LlamaAttention):
    def __init__(
        self,
        wq: nn.Module,
        wk: nn.Module,
        wv: nn.Module,
        wo: nn.Module,
        idx: int,
        args: Qwen3Config,
    ):
        assert is_flash_attn_2_available(), "Flash Attention is not available"
        super().__init__(wq, wk, wv, wo, idx, args)
        self.use_sliding_window_ = args.use_sliding_window_
        self.max_window_layers_ = args.max_window_layers_
        self.sliding_window_ = args.sliding_window_
        self.attention_dropout_ = args.attention_dropout_
        self.attention_bias_ = args.attention_bias_

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[LLMCache] = None,
    ):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_, self.head_dim_).transpose(
            1, 2
        )
        xk = xk.view(
            batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)
        xv = xv.view(
            batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        kv_seq_len = xk.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0]

        # apply rotary embedding
        cos, sin = rotary_emb
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx_) > 0
            if (
                self.sliding_window_ is not None
                and kv_seq_len > self.sliding_window_
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.sliding_window_

                past_key = past_key_value[self.layer_idx_][0]
                past_value = past_key_value[self.layer_idx_][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.sliding_window_ - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.sliding_window - 1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])],
                        dim=-1,
                    )

            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }  # Specific to RoPE models
            xk, xv = past_key_value.update(xk, xv, self.layer_idx_, cache_kwargs)

        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        input_dtype = xq.dtype
        if input_dtype == torch.float32:
            if executor.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
            xq = xq.to(target_dtype)
            xk = xk.to(target_dtype)
            xv = xv.to(target_dtype)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if (
            (self.use_sliding_window_ is None or self.use_sliding_window_)
            and self.sliding_window_ is not None
            and (
                self.max_window_layers_ is None
                or self.layer_idx_ >= self.max_window_layers_
            )
        ):
            sliding_window = self.sliding_window_
        else:
            sliding_window = None

        attn_output = flash_attention_forward(
            xq,
            xk,
            xv,
            attention_mask,
            max_seq_len,
            is_causal=self.is_causal_,
            sliding_window=sliding_window,
            dropout_p=self.attention_dropout_ if self.training else 0.0,
        ).to(input_dtype)

        attn_output = attn_output.reshape(
            batch_size, max_seq_len, self.dim_
        ).contiguous()
        attn_output = self.wo_.forward(attn_output, input_args)

        return attn_output


QWEN3_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attn": Qwen3FlashAttention,
}


class Qwen3ForCausalLM(LlamaForCausalLM):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)

    @staticmethod
    def from_pretrained(
        llm_model: modeling_qwen3.Qwen3ForCausalLM,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = executor.default_device_name(),
    ):
        llm_config: modeling_qwen3.Qwen3Config = llm_model.config
        llm_args = Qwen3Config(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
            head_dim_=llm_config.head_dim if hasattr(llm_config, "head_dim") else llm_config.hidden_size // llm_config.num_attention_heads,
            intermediate_=llm_config.intermediate_size,
            n_layers_=llm_config.num_hidden_layers,
            n_heads_=llm_config.num_attention_heads,
            n_kv_heads_=llm_config.num_key_value_heads,
            hidden_act_=llm_config.hidden_act,
            rms_norm_eps_=llm_config.rms_norm_eps,
            max_seq_len_=llm_config.max_position_embeddings,
            rope_theta_=llm_config.rope_theta,
            pad_token_id_=llm_config.pad_token_id,
            attn_implementation_=attn_impl,
            use_sliding_window_=use_sliding_window,
            sliding_window_=llm_config.sliding_window,
            max_window_layers_=llm_config.max_window_layers if hasattr(llm_config, "max_window_layers") else None,
            attention_bias_=llm_config.attention_bias if hasattr(llm_config, "attention_bias") else False,
            attention_dropout_=llm_config.attention_dropout if hasattr(llm_config, "attention_dropout") else 0.0,
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = Qwen3ForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = LlamaEmbedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_
        )
        model.norm_ = LlamaRMSNorm(llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer(idx)
            decoder.self_attn_ = QWEN3_ATTENTION_CLASSES[
                llm_args.attn_implementation_
            ](
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                idx,
                llm_args,
            )
            decoder.mlp_ = FeedForward(
                LlamaMLP(
                    layer.mlp.gate_proj,
                    layer.mlp.down_proj,
                    layer.mlp.up_proj,
                    llm_args,
                )
            )
            decoder.input_layernorm_ = LlamaRMSNorm(
                layer.input_layernorm.weight, llm_args.rms_norm_eps_
            )
            decoder.post_attention_layernorm_ = LlamaRMSNorm(
                layer.post_attention_layernorm.weight, llm_args.rms_norm_eps_
            )
            model.layers_.append(decoder)

        return model