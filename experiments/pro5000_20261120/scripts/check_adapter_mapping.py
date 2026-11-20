from pathlib import Path
import torch,json
from transformers import AutoConfig,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model
from safetensors import safe_open
P='/data/mai/benchmarks/gb256-20261112/assets/mai-base/model'
with torch.device('meta'):
 model=Qwen3VLForConditionalGeneration(AutoConfig.from_pretrained(P))
 model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM'))
expected={k.replace('.default.','.'):list(p.shape) for k,p in model.named_parameters() if 'lora_' in k}
path='/data/mai/training/mai-ui-zero2-mlp-strict10k-20261113/run/ar-adapters/ar-step-000150.safetensors'
with safe_open(path,framework='pt',device='cpu') as f:
 actual={k.replace('language_model.model.','base_model.model.model.language_model.'):list(f.get_slice(k).get_shape()) for k in f.keys()}
assert actual==expected,(list(set(actual)-set(expected))[:5],list(set(expected)-set(actual))[:5])
print(json.dumps(dict(passed=True,tensors=len(actual))))
