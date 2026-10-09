import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from flash_attn.utils.distributed import all_gather
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import argparse
import os
from utils import *
from datetime import datetime

from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType


import torch.nn.functional as F

def get_hidden_size(config):
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise ValueError("Unknown config structure")

def get_vlm_for_sequence_regression(base_model_name_or_path: str, model_name_or_path: str, model_type: str,
*,
bf16=True, lora_config=None, normalize_reward=False, use_flash_attention_2=False, ds_config: dict=None, init_value_head: bool=False, value_head_prefix="value_head",device_map=None,packing_samples=False,lora_path=None,**kwargs) -> nn.Module:
    config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
    hidden_size = get_hidden_size(config)
    print("Hidden size =", get_hidden_size(config))

    if "Qwen2" in model_name_or_path:
        from transformers import  Qwen2VLForConditionalGeneration
        cls_class = _get_reward_model(Qwen2VLForConditionalGeneration, value_head_prefix, packing_samples)
    elif "Qwen3" in model_name_or_path:
        from transformers import  Qwen3VLForConditionalGeneration
        cls_class = _get_reward_model(Qwen3VLForConditionalGeneration, value_head_prefix, packing_samples)
    else:
        print("Not Support Model!")

    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    model = cls_class.from_pretrained(model_name_or_path, config=config, trust_remote_code=True, torch_dtype = torch.bfloat16 if bf16 else "auto", quantization_config=None, device_map=device_map, **kwargs)

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
        model.eval()
    elif lora_config is not None:
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True
    
    model.config.use_cache = False

    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (hidden_size + 1))

    return model

class SoftMoEHead(nn.Module):
    """
    3-Expert soft MoE head:
      - experts: 8 separate MLPs, each maps [B,H] -> [B,1]
      - gate:    a small MLP/Linear that maps [B,H] -> [B,8], softmax to weights
      - output:  weighted sum of experts outputs
    Optional:
      - temperature: soften/sharpen gating distribution
      - return_gate: return gate weights for logging
    """
    def __init__(
        self,
        hidden_size: int,
        expert_hidden: int = 512,
        gate_hidden: int = 256,
        dropout: float = 0.0,
        temperature: float = 1.0,
        use_gate_mlp: bool = True,
    ):
        super().__init__()
        self.temperature = float(temperature)

        def make_expert():
            return nn.Sequential(
                nn.Linear(hidden_size, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, 1),
            )

        self.experts = nn.ModuleList([make_expert() for _ in range(8)])

        if use_gate_mlp:
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, gate_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, 8),
            )
        else:
            self.gate = nn.Linear(hidden_size, 8)

    def forward(self, h_last: torch.Tensor, return_gate: bool = False):
        """
        h_last: [B, H]
        returns:
          reward: [B]
          (optional) gate_w: [B,3]
        """
        # expert outputs: [B,3]
        expert_outs = torch.cat([e(h_last) for e in self.experts], dim=-1)

        # gate weights: [B,3]
        gate_logits = self.gate(h_last) / max(self.temperature, 1e-6)
        gate_w = F.softmax(gate_logits, dim=-1)

        # weighted sum: [B]
        reward = (expert_outs * gate_w).sum(dim=-1)

        return (reward, gate_w) if return_gate else reward

def _get_reward_model(base_pretrained_model, value_head_prefix="value_head", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            hidden_size = get_hidden_size(config)
            self.value_head = nn.Linear(hidden_size, 1, bias=False)
            self.packing_samples = packing_samples
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

            # self.value_head = SoftMoEHead(config.hidden_size, expert_hidden=512, gate_hidden=256, dropout=0.0, temperature=1.0)

        def forward(self, input_ids, attention_mask=None, pixel_values=None, return_output=False, **kwargs):

            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                **kwargs
            )
            # original start
            last_hidden_states = outputs.hidden_states[-1]
            values = self.value_head(last_hidden_states).squeeze(-1)
            reward = values[:, -1]
            # original ends

            # # all tokens start

            # last_hidden_states = outputs.hidden_states[-1]
            # values = self.value_head(last_hidden_states).squeeze(-1)

            # if attention_mask is None:
            #     reward = values.mean(dim=1)
            # else:
            #     mask = attention_mask.to(values.dtype) 
            #     reward = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)

            # # all tokens end


            # # moe starts
            # last_hidden_states = outputs.hidden_states[-1]  # [B,T,H]
            # idx = attention_mask.long().sum(dim=1) - 1
            # idx = idx.clamp_min(0)
            # h_last = last_hidden_states[torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), idx]  # [B,H]
            # reward = self.value_head(h_last)
            # # moe ends

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std
            return (reward, outputs) if return_output else reward

    return RewardModel


