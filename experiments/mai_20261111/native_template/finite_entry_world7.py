
import os,json,ast,textwrap
from pathlib import Path
import torch
import torch.distributed as dist
from patch_training import install
install()
import terminal_support
terminal_support.install()

import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import protocol
import mixed_prompt as mp
from mixed_native_targets import tokenize_targets
old_target=protocol.target
def convert(action):
 if 'action_type' in action:return action
 raw=old_target(action)
 obj=json.loads(raw.split('<answer>',1)[1].split('</answer>',1)[0])
 obj['action_type']={'Home':'NavigateHome','OpenApp':'OpenAPP','Enter':'KeyboardPress'}.get(obj['action_type'],obj['action_type'])
 if obj['point_2d'] is None:obj['point_2d']='None'
 if obj['action_type']=='KeyboardPress':obj['value']='Enter'
 return obj
protocol.SYSTEM=mp.SYSTEM
protocol.target=lambda a:mp.target(convert(a))
protocol.answer_fields=lambda t:json.loads(t)
protocol.parse=lambda t:dict(action='terminate' if json.loads(t)['action_type']=='Terminate' else json.loads(t)['action_type'],fields=json.loads(t))
protocol.grouped=lambda tok,texts,return_masks=False:tokenize_targets(tok,texts,return_masks)
from cosmos_framework.gui_mot import mixed_supervision as ms
ms.tokenize_grouped_targets=protocol.grouped
def context(goal,history,instruction):
 rank=int(os.environ.get('RANK','0'));level='high' if rank%2 else 'low'
 return json.dumps(dict(level=level,instruction=goal if level=='high' else instruction,history=history),ensure_ascii=False)
protocol.context=context
def messages(caption,image):
 c=json.loads(caption)
 return mp.render_policy(dict(level=c['level'],instruction=c['instruction'],current_image=image,history=[convert(x) for x in c['history']]))
protocol.messages=messages
def prefix(processor,caption,image):
 return dict(processor(text=[processor.apply_chat_template(messages(caption,image),tokenize=False,add_generation_prompt=True)],images=[image],return_tensors='pt'))
protocol.prefix=prefix
def assert_prefix(processor,inp,caption):
 c=json.loads(caption);text=processor.tokenizer.decode(inp['input_ids'][0])
 assert c['instruction'] in text
 assert ('task_goal' if c['level']=='high' else 'step_instruction') in text
 assert text.count('<|vision_start|>')==1
 return text
protocol.assert_decoded_prefix=assert_prefix

from cosmos_framework.gui_mot import native_model as nm
from cosmos_framework.gui_mot.training_artifacts import NativeTrainingArtifacts as A
from cosmos_framework.model.generator.mot import parallelize_unified_mot as pm
def full_ac(model,config):
 assert config.mode=='full'
 for name,block in list(model.model.layers.named_children()):
  model.model.layers.register_module(name,pm.apply_ac_to_module(block,config))
pm.apply_ac=full_ac
def local(p):return p.to_local() if hasattr(p,'to_local') else p
def record(name,obj):
 p=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT']);p.mkdir(parents=True,exist_ok=True)
 with (p/(name+'-rank'+str(dist.get_rank())+'.jsonl')).open('a') as f:f.write(json.dumps(obj)+chr(10))
def mask(values,m):
 assert values.shape==(1,) and dist.get_world_size()==7
 weight=0.0
 def hook(g):
  assert torch.isfinite(g).all()
  if weight==0:assert torch.count_nonzero(g)==0
  else:assert torch.count_nonzero(g)>0
  record('fm-mask-gradient',dict(mask=weight,gradient=g.detach().cpu().tolist(),passed=True))
 values.register_hook(hook)
 return values*weight
tree=ast.parse(Path(nm.__file__).read_text())
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GuiJointModel')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_compute_losses')
source=textwrap.dedent(ast.get_source_segment(Path(nm.__file__).read_text(),method))
line='vision_loss = global_sample_mean(vision_per_sample, group=group)'
assert source.count(line)==1
indent=next(x[:len(x)-len(x.lstrip())] for x in source.splitlines() if line in x)
source=source.replace(line,'vision_per_sample = mask(vision_per_sample, None)'+chr(10)+indent+line)
ns=dict(vars(nm),mask=mask);exec(source,ns);nm.GuiJointModel._compute_losses=ns['_compute_losses']
start=A.on_train_start
def begin(self,model,iteration=0):
 assert dist.get_world_size()==7 and iteration==0
 start(self,model,iteration)
 self.before={n:local(p).detach().cpu().clone() for n,p in model.net.named_parameters() if p.requires_grad and 'lora_' in n}
 torch.cuda.reset_peak_memory_stats()
A.on_train_start=begin
def gradients(self,model,iteration=0):
 counts=0
 for n,p in model.net.named_parameters():
  if p.grad is not None:
   assert torch.isfinite(local(p.grad)).all(),n
   counts+=1
 assert counts>0
 record('gradients',dict(iteration=iteration,finite=True,tensors=counts))
A.on_after_backward=gradients
end=A.on_train_end
def finish(self,model,iteration=0):
 end(self,model,iteration)
 changed={'ar':0,'fm':0}
 for n,p in model.net.named_parameters():
  if n in self.before:
   v=local(p).detach().cpu();assert torch.isfinite(v).all(),n
   changed['fm' if '_moe_gen' in n else 'ar']+=int(not torch.equal(v,self.before[n]))
 assert changed['ar']>0 and changed['fm']==0,changed
 record('complete',dict(iteration=iteration,changed=changed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,scope='unified prompt High/Low AC fixture; ALL ranks FM zero; not mixed data training',passed=True))
A.on_train_end=finish
from cosmos_framework.gui_mot.train_joint import main
main()
