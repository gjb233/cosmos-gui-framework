def stage_for(i):
 return next(rank for rank,end in enumerate([3,9,15,21,26,31,36]) if i<end)
import torch,types,dataclasses,copy,os,json
from pathlib import Path
def move(x,device):
 if isinstance(x,torch.Tensor):return x.to(device)
 if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
 if isinstance(x,tuple):
  vals=[move(v,device) for v in x]
  return type(x)(*vals) if hasattr(x,'_fields') else tuple(vals)
 if isinstance(x,list):return [move(v,device) for v in x]
 if dataclasses.is_dataclass(x) and not isinstance(x,type):
  y=copy.copy(x)
  for f in dataclasses.fields(x):object.__setattr__(y,f.name,move(getattr(x,f.name),device))
  return y
 return x
def install():
 from cosmos_framework.callbacks.norm_monitor import NormMonitor
 original_stats=NormMonitor._compute_l2_stats
 def stats(self,*args,**kwargs):
  return move(original_stats(self,*args,**kwargs),"cuda:0")
 NormMonitor._compute_l2_stats=stats
 from cosmos_framework.checkpoint import dcp as checkpoint_dcp
 original_broadcast=checkpoint_dcp._broadcast_state_dict
 def broadcast(state,rank):
  if torch.distributed.get_world_size()==1:return
  return original_broadcast(state,rank)
 checkpoint_dcp._broadcast_state_dict=broadcast
 from cosmos_framework.model.generator import omni_mot_model as om
 original=om.OmniMoTModel.build_net
 gl=dict(original.__globals__);gl['DEVICE']=om.Device.CPU
 gl['parallelize_vfm_network']=lambda net,**kwargs:net
 cpu_build=types.FunctionType(original.__code__,gl,original.__name__,original.__defaults__,original.__closure__)
 cpu_build.__kwdefaults__=original.__kwdefaults__
 def build(self,*a,**kw):
  net=cpu_build(self,*a,**kw)
  net.init_weights(buffer_device='cpu')
  self._init_lora_weights_post_materialization(net)
  return net
 om.OmniMoTModel.build_net=build
 from cosmos_framework.gui_mot.native_model import GuiJointModel
 original_to=GuiJointModel.to
 def to(self,*a,**kw):
  net=self._modules.pop('net',None)
  try:return original_to(self,*a,**kw)
  finally:
   if net is not None:self._modules['net']=net
 GuiJointModel.to=to
def place(model):
 net=model.net
 orig_forward=net.forward
 def forward(*a,**kw):
  with torch.autocast('cuda',dtype=torch.bfloat16):return orig_forward(*a,**kw)
 net.forward=forward
 assert not any('FSDP' in type(m).__name__ for m in net.modules())
 layers=net.language_model.model.layers
 assert len(layers)==36
 # Keep each stage's parameters on its card; frozen weights use the existing BF16 compute precision.
 for name,p in net.named_parameters():
  d=0
  if name.startswith('language_model.model.layers.'):d=stage_for(int(name.split('.')[3]))
  p.data=p.data.to(device='cuda:'+str(d),dtype=torch.float32 if p.requires_grad else torch.bfloat16)
 for name,b in list(net.named_buffers()):
  d=0
  if name.startswith('language_model.model.layers.'):d=stage_for(int(name.split('.')[3]))
  parent,_,key=name.rpartition('.');mod=net.get_submodule(parent) if parent else net
  mod._buffers[key]=b.to('cuda:'+str(d))
 for i,layer in enumerate(layers):
  d='cuda:'+str(stage_for(i))
  def pre(m,args,kwargs,d=d):return move(args,d),move(kwargs,d)
  layer.register_forward_pre_hook(pre,with_kwargs=True)
  original=layer.reasoner_forward
  def reason(*a,_original=original,_d=d,**kw):
   with torch.autocast('cuda',dtype=torch.bfloat16):return _original(*move(a,_d),**move(kw,_d))
  layer.reasoner_forward=reason
  # Return the last stage to the existing GPU0 output heads/loss path.
  if i==35:layer.register_forward_hook(lambda m,a,out:move(out,'cuda:0'))
 original=net.language_model.model.norm.forward
 net.language_model.model.norm.forward=lambda x:original(x.to('cuda:0'))
 original_gen=net.language_model.model.norm_moe_gen.forward
 net.language_model.model.norm_moe_gen.forward=lambda x:original_gen(x.to('cuda:0'))
 from cosmos_framework.gui_mot import multimodal as mm
 old=mm.add_deepstack
 def deep(h,v,i):
  from cosmos_framework.data.generator.sequence_packing.runtime import get_und_seq
  return old(h,move(v,get_und_seq(h).device),i)
 mm.add_deepstack=deep
 out=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT'])
 (out/'layer-placement.json').write_text(json.dumps(dict(passed=True,physical_gpus=[0,1,2,3,4,5,6],layers_per_gpu=[3,6,6,6,5,5,5],parameters_by_device={str(i):sum(p.numel() for p in net.parameters() if p.device.index==i) for i in range(7)},recomputation=False,cpu_activation_offload=False),indent=2))
