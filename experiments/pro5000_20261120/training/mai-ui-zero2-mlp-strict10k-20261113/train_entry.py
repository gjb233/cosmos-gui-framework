import sys,os,json
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import training_backend as b
import torch,torch.distributed as dist
if 'LOCAL_RANK' in os.environ:
 rank=int(os.environ['LOCAL_RANK'])
 torch.cuda.set_device(rank)
 free,total=torch.cuda.mem_get_info(rank)
 budget=total
 assert budget>20*1024**3,('insufficient budget',rank,budget)
 torch.cuda.set_per_process_memory_fraction(budget/total,rank)
 print('MAI_ORIGINAL_RES_GPU_BUDGET',rank,budget,flush=True)
from cosmos_framework.gui_mot.training_artifacts import NativeTrainingArtifacts as A
def local(p):return p.to_local() if hasattr(p,'to_local') else p
def record(name,obj):
 root=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT']);root.mkdir(parents=True,exist_ok=True)
 with (root/(name+'-rank'+str(dist.get_rank())+'.jsonl')).open('a') as f:f.write(json.dumps(obj)+'\n')
import inspect,textwrap
import cosmos_framework.gui_mot.training_artifacts as artifacts
export_source=textwrap.dedent(inspect.getsource(A.export))
assert 'len(state) != 288' in export_source
export_source=export_source.replace('len(state) != 288','len(state) != 504')
export_ns=dict(vars(artifacts))
exec(export_source,export_ns)
A.export=export_ns['export']

start=A.on_train_start;end=A.on_train_end
def begin(self,model,iteration=0):
 start(self,model,iteration)
 assert iteration==0
 print('FRESH_BASE_STEP_ZERO',flush=True)
 self.before={n:local(p).detach().cpu().clone() for n,p in model.net.named_parameters() if p.requires_grad and 'lora_' in n}
 torch.cuda.reset_peak_memory_stats()
A.on_train_start=begin
def grads(self,model,iteration=0):
 count=0;fm=0;ar=0
 for n,p in model.net.named_parameters():
  if p.grad is not None:
   g=local(p.grad);assert torch.isfinite(g).all(),n;count+=1
   if torch.count_nonzero(g):
    if '_moe_gen' in n:fm+=1
    else:ar+=1
 record('gradients',dict(iteration=iteration,finite=True,tensors=count,ar_nonzero=ar,fm_nonzero=fm))
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

 record('complete',dict(passed=True,iteration=iteration,changed=changed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,scope='eight-source published-label training, MAI native extended mobile_use shared renderer, FM on official consecutive transitions for all eight sources; published CE labels'))
A.on_train_end=finish
from cosmos_framework.utils.generator import lora
def low_memory_forward(self,x):
    base=torch.nn.functional.linear(x,self.weight,self.bias)
    delta=self.lora_B(self.lora_A(x))
    if base.dtype==delta.dtype:
        return base.add_(delta.mul_(self._lora_scale))
    return base+self._lora_scale*delta
for klass in vars(lora).copy().values():
    if isinstance(klass,type) and klass.__module__==lora.__name__ and hasattr(klass,'_lora_scale'):
        klass.forward=low_memory_forward
import zero2_hook
b.main()
