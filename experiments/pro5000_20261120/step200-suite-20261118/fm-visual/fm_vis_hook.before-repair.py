from pathlib import Path
import os,json,copy,torch,torch.distributed as dist
from PIL import Image,ImageDraw
R=Path('/data/mai/evaluation/step200-suite-20261118/fm-visual');W=Path('/data/mai/training/gui-libra-awhigh-aclow-fm2-20261117')
def blank_future(v):
 if isinstance(v,torch.Tensor):
  assert v.ndim in (4,5),v.shape
  v=v.clone()
  if v.ndim==4:v[:,1:]=0
  else:v[:,:,1:]=0
  return v
 if isinstance(v,list):return [blank_future(x) for x in v]
 raise TypeError(type(v))
def flat(value):
 while isinstance(value,list) and len(value)==1:value=value[0]
 return value
def train(self,model,dataloader_train,dataloader_val):
 rank=dist.get_rank();torch.cuda.set_device(rank)
 model=model.to('cuda',memory_format=self.config.trainer.memory_format);model.on_train_start(self.config.trainer.memory_format)
 container,scheduler=model.init_optimizer_scheduler(self.config.optimizer,self.config.scheduler)
 iteration=self.checkpointer.load(model,container,scheduler,torch.amp.GradScaler('cuda',enabled=False));assert iteration==0
 model._resume_draw_checked=True;model.eval()
 named={n:p for n,p in model.net.named_parameters() if p.requires_grad};assert len(named)==1016
 initial={n:p.detach().cpu().clone() for n,p in named.items()}
 saved=torch.load(W/'run/checkpoints/step-000200/mp_rank_00_model_states.pt',map_location='cpu',weights_only=False)['module']
 assert set(saved)=={'model.net.'+n for n in named}
 batch,stop=self._fetch_data_batch(model,iter(dataloader_train));assert not stop
 caption=json.loads(flat(batch['ai_caption']));record_id=caption['record_id']
 row=next(x for x in json.loads((R/'samples.json').read_text()) if x['id']==record_id)
 batch['video']=blank_future(batch['video'])
 samples={}
 for step,state in [(0,initial),(200,{n:saved['model.net.'+n] for n in named})]:
  with torch.no_grad():
   for n,p in named.items():p.copy_(state[n].to(p.device))
   for n,p in named.items():assert torch.equal(p.detach().cpu(),state[n])
   model._gui_prefix=None
   with torch.autocast('cuda',dtype=torch.bfloat16):
    out=model.generate_samples_from_batch(copy.deepcopy(batch),guidance=1.0,seed=[42+rank],num_steps=35)
    assert len(out['vision'])==1
    latent=out['vision'][0]
    if latent.ndim==4:latent=latent.unsqueeze(0)
    decoded=model.decode(latent)
   assert decoded.ndim==5 and torch.isfinite(decoded).all()
   pixels=((decoded[0,:,-1].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
   im=Image.fromarray(pixels);im.save(R/f'rank-{rank}-step-{step:06d}.png');samples[step]=im
   del out,latent,decoded
 panels=[Image.open(row['current_image']).convert('RGB'),Image.open(row['next_image']).convert('RGB'),samples[0],samples[200]]
 labels=['Current screenshot','Ground-truth next screenshot','Initial model (given action)','Step 200 (given action)']
 canvas=Image.new('RGB',(1280,760),'white');draw=ImageDraw.Draw(canvas)
 for i,(im,label) in enumerate(zip(panels,labels)):
  im=im.copy();im.thumbnail((312,680));canvas.paste(im,(i*320+(320-im.width)//2,55));draw.text((i*320+5,25),label,fill='black')
 canvas.save(R/f'comparison-rank-{rank}.png')
 (R/f'rank-{rank}.json').write_text(json.dumps(dict(completed=True,record_id=record_id,source=row['source'],level=caption['level'],action=row['views'][caption['level']]['action'],seed=42+rank,denoising_steps=35,conditioning='current screen + instruction/history + ground-truth action',future_pixels_zeroed=True,scope='training transition diagnostic, not held-out evaluation',checkpoint=200,panel=f'comparison-rank-{rank}.png'),ensure_ascii=False,indent=2))
 dist.barrier()
 if rank==0:
  reports=[json.loads((R/f'rank-{i}.json').read_text()) for i in range(8)];assert len({x['record_id'] for x in reports})==8;assert len({x['source'] for x in reports})==8
  html='<meta charset="utf-8"><h1>FM transition diagnostics: initial vs step 200</h1><p>Training samples; ground-truth action conditioning; no future input pixels.</p>'
  for x in reports:html+='<h2>'+x['source']+'</h2><img style="max-width:100%" src="'+x['panel']+'">'
  (R/'index.html').write_text(html);(R/'complete.json').write_text(json.dumps(dict(completed=True,samples=8)))
 dist.destroy_process_group()
