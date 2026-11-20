import sys,os,json
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
sys.path.insert(0,str(P))
import training_backend as b
import torch,torch.distributed as dist
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
if 'LOCAL_RANK' in os.environ:
 rank=int(os.environ['LOCAL_RANK'])
 torch.cuda.set_device(rank)
 free,total=torch.cuda.mem_get_info(rank)
 budget=total
 assert budget>20*1024**3,('insufficient budget',rank,budget)
 torch.cuda.set_per_process_memory_fraction(1.0,rank)
 print('MAI_ORIGINAL_RES_GPU_BUDGET',rank,budget,flush=True)
from cosmos_framework.gui_mot.training_artifacts import NativeTrainingArtifacts as A
def local(p):return p.to_local() if hasattr(p,'to_local') else p
def record(name,obj):
 root=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT']);root.mkdir(parents=True,exist_ok=True)
 with (root/(name+'-rank'+str(dist.get_rank())+'.jsonl')).open('a') as f:f.write(json.dumps(obj)+'\n')
start=A.on_train_start;end=A.on_train_end
def begin(self,model,iteration=0):
 start(self,model,iteration)
 endpoint_dir=P/'endpoint'
 def save_item(value,name):
  tmp=endpoint_dir/(name+'.pending')
  torch.save(value,tmp)
  tmp.rename(endpoint_dir/(name+'.pt'))
 # Include tied aliases under their original state-dict keys.
 endpoint_state={n:t for n,t in model.net.state_dict().items() if not n.startswith('language_model.model.layers.')}
 save_item(endpoint_state,'weights')
 del endpoint_state
 original_objective=model._compute_losses
 import inspect
 signature=inspect.signature(original_objective)
 keys=iter(range(8))
 def objective(*args,**kwargs):
  bound=signature.bind(*args,**kwargs)
  inputs=dict(bound.arguments)
  inputs.pop('out_net')
  key=next(keys)
  total,metrics=original_objective(*args,**kwargs)
  save_item(dict(arguments=inputs,fm_mask=list(model._mixed_fm_mask),expected_loss=total.detach(),
                 expected_ce=metrics['gui_ar_ce'].detach(),expected_fm=metrics['flow_matching_loss_vision'].detach()),f'objective-{key}')
  return total,metrics
 model._compute_losses=objective
 from layer_parallel import place
 place(model)
 from serial_adapter import build_serial_adapter
 adapted_forward=build_serial_adapter(model.net)
 forward_keys=iter(range(8))
 def native_adapter(*args,**kwargs):
  key=next(forward_keys)
  save_item(dict(args=args,kwargs=kwargs),f'network-input-{key}')
  with torch.autocast('cuda',dtype=torch.bfloat16):return adapted_forward(*args,**kwargs)
 model.net.forward=native_adapter
 assert iteration==0,'MAI must start fresh'
 assert os.environ['GUI_AR_ACTIVATION_CHECKPOINT']=='0'
 assert not any('CheckpointWrapper' in type(m).__name__ for m in model.net.modules())
 record('no-recompute',dict(passed=True,ar_checkpoint=False,checkpoint_wrappers=0))
 print('MAI_FRESH_ITERATION_ZERO_CONFIRMED',flush=True)
 self.before={n:local(p).detach().cpu().clone() for n,p in model.net.named_parameters() if p.requires_grad and 'lora_' in n}

 for d in range(7):torch.cuda.reset_peak_memory_stats(d)
A.on_train_start=begin
def grads(self,model,iteration=0):
 count=0;fm=0;ar=0
 for n,p in model.net.named_parameters():
  if p.grad is not None:
   g=local(p.grad);assert torch.isfinite(g).all(),n;count+=1
   if torch.count_nonzero(g):
    if '_moe_gen' in n:fm+=1
    else:ar+=1
 from activation_storage import stats
 record('gradients',dict(iteration=iteration,finite=True,tensors=count,ar_nonzero=ar,fm_nonzero=fm,peak_gib=torch.cuda.max_memory_allocated()/2**30,activation_storage=dict(stats),elapsed_since_storage_install=__import__('time').monotonic()-stats['started']))
A.on_after_backward=grads
def finish(self,model,iteration=0):
 end(self,model,iteration);changed={'ar':0,'fm':0}
 for n,p in model.net.named_parameters():
  if n in self.before:
   v=local(p).detach().cpu();assert torch.isfinite(v).all()
   changed['fm' if '_moe_gen' in n else 'ar']+=int(not torch.equal(v,self.before[n]))
 assert changed['ar']>0 and changed['fm']>0
 if iteration==1:
  root=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT'])
  sources={json.loads(l)['source'] for f in root.glob('source-packaging-*.jsonl') for l in f.read_text().splitlines()}
  expected=set(json.loads((P/'config.json').read_text())['sources'])
  assert sources==expected,('missing source in GPU validation',sources,expected)

 record('complete',dict(peaks_by_device={str(d):torch.cuda.max_memory_allocated(d)/2**30 for d in range(7)},passed=True,iteration=iteration,changed=changed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,scope='eight-source published-label training, MAI native extended mobile_use shared renderer, FM on official consecutive transitions for all eight sources; published CE labels'))
A.on_train_end=finish
b.main()
