"""Native seven-rank mixed SFT backend. Data and runtime audits remain mandatory."""
import argparse,os,json,sys,math,hashlib,ast,textwrap
from pathlib import Path
P=Path(__file__).resolve().parent
SOURCE=Path('/data/mai/code/f2f8996')
def read_rows(path):
 with open(path) as f:return [json.loads(x) for x in f if x.strip()]
def validate_inputs(config,manifest,phases):
 cfg=json.load(open(config));rows=read_rows(manifest);records={r['id']:r for r in rows}
 assert len(records)==len(rows) and records,'empty or duplicated manifest'
 assert cfg['sampling']['answer_style']=='direct_only'
 tr=cfg['train'];assert tr['world_size']==8 and tr['micro_batch_per_gpu']==8
 assert tr['global_batch']==tr['world_size']*tr['micro_batch_per_gpu']*tr['gradient_accumulation']
 assert set(r['source'] for r in rows)==set(cfg['sources']),'missing source'
 for r in rows:
  assert r.get('admission_basis')=='user_authorized_published_labels','record not admitted'
  assert r['fm_mask'] in (0,1)
  if r['fm_mask']:assert r.get('transition_verified') is True and r['next_image'] and not r['terminal']
  else:assert r['next_image'] is None
 draws=[]
 for path in phases:
  phase=read_rows(path)
  assert [x['draw'] for x in phase]==list(range(len(phase)))
  for d in phase:
   assert d['variant']=='direct' and d['id'] in records
   r=records[d['id']];assert d['source']==r['source'] and d['level'] in r['views']
  draws.extend(phase)
 assert draws
 return cfg,records,draws
def install(records,draws):
 import torch
 import torch.distributed as dist
 from PIL import Image
 import protocol
 from cosmos_framework.gui_mot import mixed_supervision as ms,native_model as nm
 from mixed_prompt import SYSTEM,render_policy,target
 from mixed_native_targets import tokenize_targets
 from mixed_native_sample import package
 protocol.SYSTEM=SYSTEM
 protocol.target=target
 from native_protocol import parse
 protocol.answer_fields=parse
 protocol.grouped=lambda tok,texts,return_masks=False:tokenize_targets(tok,texts,return_masks)
 ms.tokenize_grouped_targets=protocol.grouped
 def prefix(processor,caption,image):
  c=json.loads(caption)
  policy=dict(current_image=image,level=c['level'],instruction=c['instruction'],history=c['history'],goal=c['goal'])
  messages=render_policy(policy)
  messages[1]['content'].reverse()
  w,h=image.size
  inp=dict(processor(text=[processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)],images=[image],return_tensors='pt',images_kwargs={'size':{'shortest_edge':1024,'longest_edge':max(16777216,math.ceil(h/32)*32*math.ceil(w/32)*32)}}))
  assert inp['input_ids'].shape[-1]<=10000, ('input token limit',inp['input_ids'].shape[-1])
  gh,gw=map(int,inp['image_grid_thw'][0][-2:]);ph=gh*16;pw=gw*16
  assert abs(ph-h)<=32 and abs(pw-w)<=32,(h,w,ph,pw)
  return inp
 protocol.prefix=prefix
 def check_prefix(processor,inp,caption):
  c=json.loads(caption);text=processor.tokenizer.decode(inp['input_ids'][0])
  policy=dict(current_image=None,image_size=c['image_size'],level=c['level'],instruction=c['instruction'],history=c['history'],goal=c['goal'])
  expected=render_policy(policy)[1]['content'][0]['text']
  assert expected in text and text.count('<|vision_start|>')==1
  assert '<thinking>' not in expected
  assert '<thinking>' not in text.split('<|im_start|>assistant')[-1]
  return text
 protocol.assert_decoded_prefix=check_prefix
 class Dataset:
  def __init__(self,*args,**kwargs):pass
  def __len__(self):return len(draws)-int(os.environ.get('GUI_START_DRAW','0'))
  def __getitem__(self,index):
   index+=int(os.environ.get('GUI_START_DRAW','0'))
   assert index<len(draws),index
   draw=draws[index];r=records[draw['id']];level=draw['level']
   sample=package(r,level)
   sample['ai_caption']=json.dumps(dict(draw_slot=index,slot_weight=1.,level=level,goal=r['goal'],instruction=r['views'][level]['instruction'],
     image_size=list(reversed(sample['gui_current_screens'][0].shape[-2:])),history=r['history'],fm_mask=r['fm_mask'],record_id=r['id']),ensure_ascii=False)
   return sample
 ms.MixedJointPolicyDataset=Dataset
 previous=nm.GuiJointModel._prepare_gui_prefix
 def prepare(self,batch):
  captions=batch['ai_caption']
  while isinstance(captions,list) and len(captions)==1 and isinstance(captions[0],list):captions=captions[0]
  if isinstance(captions,str):captions=[captions]
  if not hasattr(self,'_resume_draw_checked'):
   import torch.distributed as dist
   rank=dist.get_rank(); actual=json.loads(captions[0]); expected=draws[rank+int(os.environ.get('GUI_START_DRAW','0'))]
   assert (actual['record_id'],actual['level'])==(expected['id'],expected['level']),('resume draw mismatch',rank,actual['record_id'],expected['id'])
   self._resume_draw_checked=True
  self._slot_weight=json.loads(captions[0])['slot_weight']
  self._mixed_fm_mask=[json.loads(x)['fm_mask'] for x in captions]
  return previous(self,batch)
 nm.GuiJointModel._prepare_gui_prefix=prepare
 sourcefile=Path(nm.__file__).read_text();tree=ast.parse(sourcefile)
 cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GuiJointModel')
 method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_compute_losses')
 code=textwrap.dedent(ast.get_source_segment(sourcefile,method))
 line='vision_loss = global_sample_mean(vision_per_sample, group=group)'
 assert code.count(line)==1
 def mask(values,weights):
  m=torch.as_tensor(weights,dtype=values.dtype,device=values.device)
  assert m.shape==values.shape and ((m==0)|(m==1)).all()
  return values*m
 code=code.replace(line,'vision_loss = global_sample_mean(mixed_mask(vision_per_sample, self._mixed_fm_mask)*self._slot_weight, group=group)')
 code=code.replace('self.hybrid_ce_weight * ar_loss', 'self.hybrid_ce_weight * (ar_loss * self._slot_weight)')
 loss_cfg=json.loads((P/'config.json').read_text())['loss']
 assert loss_cfg['ce_weight']==1.0 and loss_cfg['fm_weight']==2.0
 assert code.count('rf.loss_scale * vision_loss')==2
 code=code.replace('rf.loss_scale * vision_loss','2.0 * rf.loss_scale * vision_loss')
 code=code.replace('rf = self.config.rectified_flow_training_config','rf = self.config.rectified_flow_training_config\n        assert rf.loss_scale == 1.0 and self.hybrid_ce_weight == 1.0')
 code=code.replace('return total, {','audit_fm_weights(self,total,ar_loss,vision_loss)\n        return total, {')
 def audit_fm_weights(model,total,ce,fm):
  assert model.video_only and model.mot_joint and model.hybrid_kd_weight==0
  assert torch.allclose(total.detach(),(ce+2.0*fm).detach(),rtol=1e-5,atol=1e-6)
  if not getattr(model,'_fm2_audited',False):
   import torch.distributed as dist
   report=dict(ce_weight=1.0,fm_weight=2.0,ce=float(ce.detach()),fm_unweighted=float(fm.detach()),total=float(total.detach()),passed=True)
   (Path(os.environ['GUI_ADAPTER_EXPORT_ROOT'])/f'loss-weights-rank{dist.get_rank()}.json').write_text(json.dumps(report))
   model._fm2_audited=True
 ns=dict(vars(nm),mixed_mask=mask,audit_fm_weights=audit_fm_weights);exec(code,ns)
 nm.GuiJointModel._compute_losses=ns['_compute_losses']
 from cosmos_framework.model.generator.mot import parallelize_unified_mot as pm
 def full_ac(model,config):
  assert config.mode=='full';return
  for name,block in list(model.model.layers.named_children()):
   model.model.layers.register_module(name,pm.apply_ac_to_module(block,config))
 pm.apply_ac=full_ac
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--config',required=True);parser.add_argument('--manifest',required=True)
 parser.add_argument('--phase1',required=True);parser.add_argument('--phase2',required=True)
 parser.add_argument('--native-config',default=str(P/'train-mixed-native.toml'))
 parser.add_argument('--validate-only',action='store_true')
 parser.add_argument('--dryrun',action='store_true')
 args=parser.parse_args()
 cfg,records,draws=validate_inputs(args.config,args.manifest,[args.phase1,args.phase2])
 if args.validate_only:
  print(json.dumps(dict(records=len(records),draws=len(draws),global_batch=cfg['train']['global_batch'])));return
 assert args.dryrun or int(os.environ.get('WORLD_SIZE','0'))==cfg['train']['world_size']
 assert Path(args.native_config).is_file()
 os.environ.update(GUI_TRAIN_MANIFEST=str(Path(args.manifest).resolve()),GUI_RESPONSE_STYLE='direct',GUI_GROUPED_CE='1',
  GUI_HORIZON='1',GUI_INSTRUCTION_LEVEL='mixed',GUI_MOT_JOINT='1',GUI_HYBRID_AR='0',
  GUI_NORMALIZE_PLAN='0',GUI_AR_ACTIVATION_CHECKPOINT='0',GUI_DTENSOR_SAFE_INIT='1',
  GUI_FSDP_MASTER_DTYPE='float32',GUI_BACKBONE_PATH='/data/mai/benchmarks/gui-libra-base/ready-model',
  WAN_VAE_PATH='/data/mai/benchmarks/gb256-20261112/assets/wan-vae/Wan2.2_VAE.pth',
  BASE_CHECKPOINT_PATH='/data/mai/benchmarks/gb256-20261112/assets/cosmos-base/Cosmos3-Nano-DCP')
 # Runtime installation only uses the isolated, already verified loader/vision fixes.
 sys.path.insert(0,str(SOURCE));sys.path.insert(0,'/data/mai/code/f2f8996/experiments/mai_20261111/native_template');sys.path.insert(0,str(P))
 from patch_training import install as install_runtime
 install_runtime()
 install(records,draws[:185600])
 sys.argv=[sys.argv[0],'--sft-toml',args.native_config,'--','trainer.seed=42','job.wandb_mode=disabled','dataloader_train.max_samples_per_batch=8']
 sys.argv.append('model.config.lora_target_modules="q_proj,k_proj,v_proj,o_proj,q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen,mlp.gate_proj,mlp.up_proj,mlp.down_proj,mlp_moe_gen.gate_proj,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj"')
 if cfg['train']['schedule']=='linear_decay':sys.argv.append('scheduler.lr_scheduler_type=lambdalinear')
 if args.dryrun:sys.argv.insert(1,'--dryrun')
 from cosmos_framework.gui_mot.train_joint import main as native_main
 native_main()
if __name__=='__main__':main()
