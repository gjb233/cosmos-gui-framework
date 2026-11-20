import os,json,time,math,random
from pathlib import Path
import numpy as np
import torch,torch.distributed as dist
import deepspeed
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model
from PIL import Image
from native_protocol import render_policy,target
from mixed_native_targets import tokenize_targets
W=Path(__file__).parent;R=W/'run';A=R/'audit'
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
deepseed=42
random.seed(deepseed);np.random.seed(deepseed);torch.manual_seed(deepseed);torch.cuda.manual_seed_all(deepseed)
torch.set_num_threads(2)
base='/data/mai/benchmarks/gb256-20261112/assets/mai-base/model'
processor=AutoProcessor.from_pretrained(base,local_files_only=True)
records={r['id']:r for r in map(json.loads,(W/'train-manifest.jsonl').read_text().splitlines())}
draws=[]
for name in ['phase1.jsonl','phase2.jsonl']:draws.extend(map(json.loads,(W/name).read_text().splitlines()))
draws=draws[:185600];assert len(records)==61864 and len(draws)==725*256
model=Qwen3VLForConditionalGeneration.from_pretrained(base,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0,bias='none',target_modules=r'.*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))',task_type='CAUSAL_LM'))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
named={n:p for n,p in model.named_parameters() if p.requires_grad};assert len(named)==504,len(named)
for p in named.values():p.data=p.data.float()
before={n:p.detach().cpu().clone() for n,p in named.items()}
(A/f'trainable-rank{rank}.json').write_text(json.dumps({'names':list(named),'tensors':len(named),'parameters':sum(p.numel() for p in named.values())}))
def encode(slot):
 d=draws[slot];r=records[d['id']];level=d['level'];v=r['views'][level]
 with Image.open(r['current_image']) as f:im=f.convert('RGB')
 messages=render_policy(dict(current_image=im,level=level,instruction=v['instruction'],history=r['history'],goal=r['goal']));messages[1]['content'].reverse()
 text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 x=dict(processor(text=[text],images=[im],return_tensors='pt',images_kwargs={'size':{'shortest_edge':1024,'longest_edge':max(16777216,math.ceil(im.height/32)*32*math.ceil(im.width/32)*32)}}))
 ids=tokenize_targets(processor.tokenizer,[target(v['action'],level)])[0]
 assert x['input_ids'].shape[1]<=10000
 n=x['input_ids'].shape[1];x['input_ids']=torch.cat([x['input_ids'],torch.tensor([ids[:-1]])],dim=1);x['attention_mask']=torch.ones_like(x['input_ids'])
 assert processor.tokenizer.decode(x['input_ids'][0,:n]).rstrip().endswith('<|im_start|>assistant')
 return x,torch.tensor([ids]),dict(slot=slot,id=d['id'],level=level,prefix_tokens=n,target_tokens=len(ids))
class Objective(torch.nn.Module):
 def __init__(self,m):super().__init__();self.vlm=m
 def forward(self,slots):
  losses=[];audit=[]
  for slot in slots:
   x,y,info=encode(slot);x={k:v.cuda() for k,v in x.items()};y=y.cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):
    out=self.vlm(**x,logits_to_keep=y.shape[1],use_cache=False)
    assert out.logits.shape[:2]==y.shape
    loss=torch.nn.functional.cross_entropy(out.logits.float().reshape(-1,out.logits.shape[-1]),y.reshape(-1))
   losses.append(loss);audit.append(info)
  self.audit=audit
  return torch.stack(losses).mean()
optimizer=torch.optim.AdamW(list(named.values()),lr=1e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
config=dict(train_batch_size=256,train_micro_batch_size_per_gpu=8,gradient_accumulation_steps=4,zero_optimization=dict(stage=2,overlap_comm=False,contiguous_gradients=True,reduce_bucket_size=50000000,allgather_bucket_size=50000000),gradient_clipping=1.,steps_per_print=100000,zero_allow_untested_optimizer=True)
engine,_,_,_=deepspeed.initialize(model=Objective(model),optimizer=optimizer,model_parameters=list(named.values()),config=config)
engine.train()
for step in range(725):
 lr=1e-5*(step/7 if step<7 else (724-step)/717)
 for group in engine.optimizer.param_groups:group['lr']=lr
 for group in engine.optimizer.optimizer.param_groups:group['lr']=lr
 dist.barrier();torch.cuda.synchronize();t=time.monotonic();losses=[]
 for micro in range(4):
  slots=[rank+8*((step*4+micro)*8+i) for i in range(8)]
  loss=engine(slots);assert torch.isfinite(loss)
  engine.backward(loss);engine.step();losses.append(loss.detach())
  if step<2:
   with (A/f'first-draws-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(engine.module.audit)+'\n')
 torch.cuda.synchronize();elapsed=torch.tensor(time.monotonic()-t,device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
 avg=torch.stack(losses).mean();dist.all_reduce(avg);avg/=8
 row=dict(step=step+1,seconds=elapsed.item(),loss=avg.item(),lr=lr,global_batch=256)
 with (A/f'timing-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if rank==0:print('STEP',row,flush=True)
 if step==1 or (step+1)%50==0 or step==724:
  assert all(torch.isfinite(p).all() for p in named.values())
  tag=f'step-{step+1:06d}'
  engine.save_checkpoint(str(R/'checkpoints'),tag=tag,client_state={'completed_steps':step+1},exclude_frozen_parameters=True)
  torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()),R/'checkpoints'/tag/f'rng-rank{rank}.pt')
  if rank==0:
   model.save_pretrained(R/'adapters'/tag);processor.save_pretrained(R/'adapters'/tag)
   (R/'checkpoints/latest-complete.json').write_text(json.dumps(dict(completed_steps=step+1,tag=tag)))
  dist.barrier()
  if step==1:
   changed=[n for n,p in named.items() if not torch.equal(p.detach().cpu(),before[n])]
   assert any('.mlp.' in n for n in changed) and any('.self_attn.' in n for n in changed)
   (A/f'startup-rank{rank}.json').write_text(json.dumps(dict(passed=True,changed=len(changed),mlp_changed=sum('.mlp.' in n for n in changed))))
(A/f'complete-rank{rank}.json').write_text(json.dumps(dict(passed=True,steps=725)))
dist.barrier();dist.destroy_process_group()
