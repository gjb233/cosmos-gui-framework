"""Native seven-rank mixed SFT backend. Data and runtime audits remain mandatory."""
import argparse,os,json,sys,math,hashlib,ast,textwrap
from pathlib import Path
P=Path(__file__).resolve().parent
SOURCE=Path('/rscratch/wzzheng/experiments/gui-grounding-fixed-ab-20261002-v1/source')
def read_rows(path):
 with open(path) as f:return [json.loads(x) for x in f if x.strip()]
def validate_inputs(config,manifest,phases):
 cfg=json.load(open(config));rows=read_rows(manifest);records={r['id']:r for r in rows}
 assert len(records)==len(rows) and records,'empty or duplicated manifest'
 assert cfg['sampling']['answer_style']=='direct_only'
 tr=cfg['train'];assert tr['world_size']==7 and tr['micro_batch_per_gpu']==1
 assert tr['global_batch']==tr['world_size']*tr['micro_batch_per_gpu']*tr['gradient_accumulation']
 assert set(r['source'] for r in rows)==set(cfg['sources']),'missing source'
 for r in rows:
  assert r.get('approved_for_training') is True,'record not approved'
  assert r['fm_mask'] in (0,1)
  if r['fm_mask']:assert r.get('action_alignment_verified') is True and r['next_image'] and not r['terminal']
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
 protocol.answer_fields=lambda t:json.loads(t)
 protocol.grouped=lambda tok,texts,return_masks=False:tokenize_targets(tok,texts,return_masks)
 ms.tokenize_grouped_targets=protocol.grouped
 def prefix(processor,caption,image):
  c=json.loads(caption)
  policy=dict(current_image=image,level=c['level'],instruction=c['instruction'],history=c['history'])
  messages=render_policy(policy)
  return dict(processor(text=[processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)],images=[image],return_tensors='pt'))
 protocol.prefix=prefix
 def check_prefix(processor,inp,caption):
  c=json.loads(caption);text=processor.tokenizer.decode(inp['input_ids'][0])
  policy=dict(current_image=None,level=c['level'],instruction=c['instruction'],history=c['history'])
  expected=render_policy(policy)[1]['content'][1]['text']
  assert expected in text and text.count('<|vision_start|>')==1
  payload=json.loads(expected)
  assert set(payload)=={'instruction_level','previous_actions','step_instruction' if c['level']=='low' else 'task_goal'}
  return text
 protocol.assert_decoded_prefix=check_prefix
 class Dataset:
  def __init__(self,*args,**kwargs):pass
  def __len__(self):return len(draws)
  def __getitem__(self,index):
   draw=draws[index];r=records[draw['id']];level=draw['level']
   sample=package(r,level)
   sample['ai_caption']=json.dumps(dict(level=level,instruction=r['views'][level]['instruction'],
     history=r.get('history',[]),fm_mask=r['fm_mask'],record_id=r['id']),ensure_ascii=False)
   return sample
 ms.MixedJointPolicyDataset=Dataset
 previous=nm.GuiJointModel._prepare_gui_prefix
 def prepare(self,batch):
  captions=batch['ai_caption']
  while isinstance(captions,list) and len(captions)==1 and isinstance(captions[0],list):captions=captions[0]
  if isinstance(captions,str):captions=[captions]
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
 code=code.replace(line,'vision_loss = global_sample_mean(mixed_mask(vision_per_sample, self._mixed_fm_mask), group=group)')
 ns=dict(vars(nm),mixed_mask=mask);exec(code,ns)
 nm.GuiJointModel._compute_losses=ns['_compute_losses']
 from cosmos_framework.model.generator.mot import parallelize_unified_mot as pm
 def full_ac(model,config):
  assert config.mode=='full'
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
 assert args.dryrun or int(os.environ.get('WORLD_SIZE','0'))==7
 assert Path(args.native_config).is_file()
 os.environ.update(GUI_TRAIN_MANIFEST=str(Path(args.manifest).resolve()),GUI_RESPONSE_STYLE='direct',GUI_GROUPED_CE='1',
  GUI_HORIZON='1',GUI_INSTRUCTION_LEVEL='mixed',GUI_MOT_JOINT='1',GUI_HYBRID_AR='0',
  GUI_NORMALIZE_PLAN='0',GUI_AR_ACTIVATION_CHECKPOINT='1',GUI_DTENSOR_SAFE_INIT='1',
  GUI_FSDP_MASTER_DTYPE='float32',GUI_BACKBONE_PATH='/rscratch/wzzheng/code/cosmos-gui/.local/models/GUI-Libra-8B',
  WAN_VAE_PATH='/rscratch/wzzheng/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth',
  BASE_CHECKPOINT_PATH='/rscratch/wzzheng/code/cosmos-gui/.local/models/Cosmos3-Nano-DCP')
 # Runtime installation only uses the isolated, already verified loader/vision fixes.
 sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(P/'unified-template-debug'))
 from patch_training import install as install_runtime
 install_runtime()
 install(records,draws)
 sys.argv=[sys.argv[0],'--sft-toml',args.native_config,'--','trainer.seed=42','job.wandb_mode=disabled']
 if cfg['train']['schedule']=='linear_decay':sys.argv.append('scheduler.lr_scheduler_type=lambdalinear')
 if args.dryrun:sys.argv.insert(1,'--dryrun')
 from cosmos_framework.gui_mot.train_joint import main as native_main
 native_main()
if __name__=='__main__':main()
