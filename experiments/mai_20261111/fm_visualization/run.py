import os,json,subprocess,fcntl,socket,time
from pathlib import Path
o=Path(__file__).resolve().parent;p=o.parent
t=p/'mai-ui-70283-resume150-fsdp7-micro4-gb56-a22-fullmem'
source='/rscratch/wzzheng/experiments/gui-grounding-fixed-ab-20261002-v1/source'
py='/rscratch/wzzheng/code/cosmos-gui/.local/gui-venv/bin/python'
assert socket.gethostname().split('.')[0]=='a23'
locks=[]
for i in [6,7]:
 row=subprocess.check_output(['nvidia-smi','-i',str(i),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert int(row[1])>26000,row
 f=open('/tmp/mixed-eight-eval-'+row[0].strip()+'.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
while not (o/'selection.json').exists():time.sleep(10)
for step in [100,200,300]:
 env={k:v for k,v in os.environ.items() if not k.startswith(('GUI_','WANDB_'))}
 env.update(LD_LIBRARY_PATH='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=source,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID',
 GUI_SUPERVISION_CACHE=str(p/'fm-debug-draft/direct-cache.json'),GUI_TRAIN_MANIFEST=str(t/'artifacts/train-manifest.jsonl'),GUI_RESPONSE_STYLE='direct',GUI_GROUPED_CE='1',GUI_HORIZON='1',GUI_INSTRUCTION_LEVEL='mixed',GUI_MOT_JOINT='1',GUI_HYBRID_AR='0',GUI_NORMALIZE_PLAN='0',GUI_AR_ACTIVATION_CHECKPOINT='0',GUI_DTENSOR_SAFE_INIT='1',GUI_FSDP_MASTER_DTYPE='bfloat16',
 GUI_BACKBONE_PATH='/rscratch/wzzheng/experiments/gui-mai-lowres-abd-a23-20261005-v1/model',WAN_VAE_PATH='/rscratch/wzzheng/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth',BASE_CHECKPOINT_PATH='/rscratch/wzzheng/code/cosmos-gui/.local/models/Cosmos3-Nano-DCP',
 GUI_RUN_NAME=f'fm-test-{step}',IMAGINAIRE_OUTPUT_ROOT=str(o/'runtime'),GUI_ADAPTER_EXPORT_ROOT=str(o/'instrument'),FM_STEP=str(step),PYTORCH_ALLOC_CONF='expandable_segments:True')
 env.pop('MOT_CHECKPOINT_PATH',None)
 (o/'instrument').mkdir(exist_ok=True)
 cmd=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(o/'entry.py'),'--sft-toml',str(o/'inference.toml'),'--','job.wandb_mode=disabled','trainer.seed=42','trainer.max_iter=0','checkpoint.load_training_state=False','trainer.cudnn.benchmark=False','trainer.cudnn.deterministic=True']
 with (o/f'run-{step}.log').open('a') as log:
  child=subprocess.Popen(cmd,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (o/'active.json').write_text(json.dumps(dict(pid=child.pid,controller_pid=os.getpid(),step=step,command=cmd,gpus=[6,7])))
  rc=child.wait()
 (o/f'exit-{step}.json').write_text(json.dumps(dict(exit_code=rc)))
 if rc:raise RuntimeError((step,rc))
(o/'complete.json').write_text(json.dumps(dict(steps=[100,200,300],samples=4)))
