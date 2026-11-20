import dataclasses,json,torch
from pathlib import Path
def describe(x):
 if isinstance(x,torch.Tensor):return dict(kind='tensor',shape=list(x.shape),dtype=str(x.dtype),device=str(x.device),requires_grad=x.requires_grad)
 if dataclasses.is_dataclass(x):return dict(kind='dataclass',name=type(x).__module__+'.'+type(x).__qualname__,fields={f.name:describe(getattr(x,f.name)) for f in dataclasses.fields(x)})
 if isinstance(x,dict):return {str(k):describe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return dict(kind=type(x).__name__,items=[describe(v) for v in x])
 if x is None or isinstance(x,(int,float,str,bool)):return x
 return dict(kind='unsupported',name=type(x).__module__+'.'+type(x).__qualname__)
def install(model,root):
 root=Path(root);counts={}
 def record(i,path,args,kwargs):
  key=(i,path);n=counts.get(key,0);counts[key]=n+1
  if n>=8:return
  with (root/'boundary-schemas.jsonl').open('a') as f:f.write(json.dumps(dict(layer=i,path=path,microbatch_index=n,args=describe(args),kwargs=describe(kwargs)))+'\n')
 for i in [0,7,17,26,35]:
  layer=model.net.language_model.model.layers[i]
  def pre(m,args,kwargs,i=i):record(i,'joint_fm',args,kwargs)
  layer.register_forward_pre_hook(pre,with_kwargs=True)
  old=layer.reasoner_forward
  def forward(*args,_old=old,_i=i,**kwargs):
   record(_i,'causal_ar',args,kwargs)
   return _old(*args,**kwargs)
  layer.reasoner_forward=forward
