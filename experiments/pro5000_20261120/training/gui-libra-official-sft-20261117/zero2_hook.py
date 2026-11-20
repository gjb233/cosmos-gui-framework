from zero2_micro2_adapter import audit_batch,audit_outputs
import os,json,time,torch,random,numpy as np
import torch.distributed as dist
from pathlib import Path
from torch.utils.checkpoint import checkpoint
import cosmos_framework.model.generator.omni_mot_model as om
from cosmos_framework.trainer import ImaginaireTrainer as Trainer

def replicated(net,parallel_dims,compile_config,ac_config,attention_io_layout,mp_policy):
 assert ac_config.mode=='full' and not compile_config.enabled
 net.attention_io_layout=attention_io_layout
 for p in net.parameters():p.data=p.data.to(torch.float32 if p.requires_grad else torch.bfloat16)
 return net
om.parallelize_vfm_network=replicated
class LossModule(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,batch,iteration):
  with torch.autocast('cuda',dtype=torch.bfloat16):
   output,loss=self.model.training_step(batch,iteration)
  self.output=output;return output.get('_backward_loss',loss)

def train(self,model,dataloader_train,dataloader_val):
 import deepspeed
 root=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT']);root.mkdir(exist_ok=True,parents=True)
 model=model.to('cuda',memory_format=self.config.trainer.memory_format);model.on_train_start(self.config.trainer.memory_format)
 container,scheduler=model.init_optimizer_scheduler(self.config.optimizer,self.config.scheduler)
 scaler=torch.amp.GradScaler('cuda',enabled=False)
 iteration=self.checkpointer.load(model,container,scheduler,scaler);assert iteration==0
 self.callbacks.on_train_start(model,iteration=0)
 named={n:p for n,p in model.net.named_parameters() if p.requires_grad}
 trainable=list(named.values());assert len(trainable)==1016
 for p in trainable:p.data=p.data.float()
 before={n:p.detach().cpu().clone() for n,p in named.items()}
 mlp={n:p for n,p in named.items() if 'lora_' in n and ('.mlp.' in n or '.mlp_moe_gen.' in n)}
 assert len(mlp)==432,len(mlp)
 (root/f'trainable-rank{dist.get_rank()}.json').write_text(json.dumps(dict(total_parameters=sum(p.numel() for p in named.values()),mlp_parameters=sum(p.numel() for p in mlp.values()),names=list(named))))
 counts={k:dict(requested=0,executed=0) for k in ('fm','ar')}
 def recompute(fn,branch):
  def body(*args,**kwargs):
   counts[branch]['executed']+=1
   return fn(*args,**kwargs)
  def forward(*args,**kwargs):
   counts[branch]['requested']+=1
   return checkpoint(body,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
  return forward
 for layer in model.net.language_model.model.layers:
  layer.forward=recompute(layer.forward,'fm')
  layer.reasoner_forward=recompute(layer.reasoner_forward,'ar')
 assert len(model.net.language_model.model.layers)==36
 config=dict(train_batch_size=256,train_micro_batch_size_per_gpu=8,gradient_accumulation_steps=4,zero_optimization=dict(stage=2,overlap_comm=False,contiguous_gradients=True,reduce_bucket_size=50000000,allgather_bucket_size=50000000),gradient_clipping=1.0,steps_per_print=100000,zero_allow_untested_optimizer=True)
 optimizer,=container.optimizers
 engine,_,_,_=deepspeed.initialize(model=LossModule(model),optimizer=optimizer,model_parameters=trainable,config=config)
 engine.train()
 assert engine.zero_optimization_stage()==2 and engine.train_batch_size()==256
 (root/f'effective-rank{dist.get_rank()}.json').write_text(json.dumps(dict(config=config,trainable_tensors=len(trainable),trainable_dtype='float32',activation_checkpointing='non-reentrant, all 36 layers, both FM and AR paths')))
 run= root.parent
 plan=json.loads((run.parent/'training-plan.json').read_text())
 total=plan['optimizer_steps'];warmup=plan['warmup_steps']
 start_step=int(os.environ.get('GUI_START_DRAW','0'))//256
 checkpoint_root=run/'checkpoints'
 if start_step:
  tag=f'step-{start_step:06d}'
  path,state=engine.load_checkpoint(str(checkpoint_root),tag=tag,load_module_strict=False)
  assert path and state['completed_steps']==start_step and state['plan']==plan
  rng=torch.load(checkpoint_root/tag/f'rng-rank{dist.get_rank()}.pt',weights_only=False,map_location='cpu')
  assert engine.global_steps==start_step
  saved=torch.load(checkpoint_root/tag/'mp_rank_00_model_states.pt',map_location='cpu',weights_only=False)['module']
  assert set(saved)=={'model.net.'+n for n in named}
  for n,p in named.items():assert torch.equal(p.detach().cpu(),saved['model.net.'+n]),n
  before={n:saved['model.net.'+n].clone() for n in named};del saved
  saved_opt=torch.load(checkpoint_root/tag/f'zero_pp_rank_{dist.get_rank()}_mp_rank_00_optim_states.pt',map_location='cpu',weights_only=False)['optimizer_state_dict']['base_optimizer_state']['state']
  loaded_opt=engine.optimizer.optimizer.state_dict()['state'];assert set(loaded_opt)==set(saved_opt)
  for k,values in saved_opt.items():
   assert set(values)==set(loaded_opt[k])
   for key,value in values.items():
    actual=loaded_opt[k][key]
    assert torch.equal(torch.as_tensor(actual).cpu(),torch.as_tensor(value).cpu()),(k,key)
   assert int(values['step'])==start_step
  (root/f'resume-{start_step}-rank{dist.get_rank()}.json').write_text(json.dumps(dict(passed=True,checkpoint_step=start_step,trainable_tensors=len(named),weights_exact=True,adam_exact=True,next_draw=start_step*256+dist.get_rank())))
  del saved_opt,loaded_opt
 def save(completed):
  tag=f'step-{completed:06d}'
  engine.save_checkpoint(str(checkpoint_root),tag=tag,client_state=dict(completed_steps=completed,plan=plan),save_latest=False,exclude_frozen_parameters=True)
  torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()),checkpoint_root/tag/f'rng-rank{dist.get_rank()}.pt')
  assert all(torch.isfinite(p).all() for p in named.values())
  from cosmos_framework.gui_mot.training_artifacts import NativeTrainingArtifacts
  exporter=NativeTrainingArtifacts(str(run/'ar-adapters'),every_n=1);exporter.export(model,completed)
  dist.barrier()
  if dist.get_rank()==0:
   files=list((checkpoint_root/tag).glob('*optim_states.pt'))
   assert len(files)==8,(tag,len(files))
   p=checkpoint_root/'latest-complete.json';tmp=p.with_suffix('.tmp')
   tmp.write_text(json.dumps(dict(tag=tag,completed_steps=completed,plan=plan)));tmp.replace(p)
   print('CHECKPOINT_SAVED',tag,flush=True)
  dist.barrier()
 if hasattr(dataloader_train,'set_start_iteration'):dataloader_train.set_start_iteration(0)
 it=iter(dataloader_train)
 if start_step:
  random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
 for step in range(start_step,total):
  lr=1e-5*(step/warmup if step<warmup else (total-1-step)/max(1,total-1-warmup))
  for group in engine.optimizer.param_groups:group['lr']=lr
  for group in engine.optimizer.optimizer.param_groups:group['lr']=lr
  dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic();losses=[]
  for micro in range(4):
   batch,stop=self._fetch_data_batch(model,it);assert not stop
   audit_batch(model,batch,step,micro)
   loss=engine(batch,step);assert torch.isfinite(loss).all()
   audit_outputs(model,step,micro)
   engine.backward(loss);engine.step();losses.append(float(loss.detach()))
   if micro==0:
    assert all(v['executed']>v['requested'] for v in counts.values()),counts
    (root/f'recompute-rank{dist.get_rank()}.json').write_text(json.dumps(counts))
  torch.cuda.synchronize();elapsed=torch.tensor(time.monotonic()-start,device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
  assert engine.global_steps==step+1
  row=dict(step=step+1,seconds=elapsed.item(),samples_per_second=256/elapsed.item(),loss=sum(losses)/4,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,global_batch=256,optimizer_updates=engine.global_steps,lr=lr,recomputation=counts)
  with (root/f'timing-rank{dist.get_rank()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print('ZERO2_TRAIN_STEP',row,flush=True)
  if start_step and step==start_step:
   updated={branch:sum(not torch.equal(p.detach().cpu(),before[n]) for n,p in mlp.items() if ('_moe_gen' in n)==(branch=='fm')) for branch in ('ar','fm')}
   assert updated['ar']>0 and updated['fm']>0,updated
   (root/f'resume-update-{start_step}-rank{dist.get_rank()}.json').write_text(json.dumps(dict(step=step+1,mlp_updated=updated,finite=all(torch.isfinite(p).all().item() for p in named.values()))))
  if step+1==2 or (step+1)%50==0 or step+1==total or (start_step and step==start_step):
   if step+1==2:
    updates={branch:sum(not torch.equal(p.detach().cpu(),before[n]) for n,p in mlp.items() if ('_moe_gen' in n)==(branch=='fm')) for branch in ('ar','fm')}
    assert updates==dict(ar=108,fm=108),updates
    (root/f'early-mlp-updates-rank{dist.get_rank()}.json').write_text(json.dumps(updates))
   save(step+1)
 changed=dict(ar=0,fm=0,other=0)
 for n,p in named.items():
  value=p.detach().cpu();assert torch.isfinite(value).all(),n
  branch='fm' if '_moe_gen' in n else 'ar' if 'lora_' in n else 'other'
  changed[branch]+=int(not torch.equal(value,before[n]))
 assert changed==dict(ar=504,fm=504,other=8),changed
 mlp_changed={branch:sum(not torch.equal(p.detach().cpu(),before[n]) for n,p in mlp.items() if ('_moe_gen' in n)==(branch=='fm')) for branch in ('ar','fm')}
 assert mlp_changed==dict(ar=216,fm=216),mlp_changed
 (root/f'mlp-updates-rank{dist.get_rank()}.json').write_text(json.dumps(mlp_changed))
 (root/f'complete-rank{dist.get_rank()}.json').write_text(json.dumps(dict(passed=True,optimizer_steps=engine.global_steps,changed=changed,recomputation=counts)))
 dist.barrier();dist.destroy_process_group()
Trainer.train=train
