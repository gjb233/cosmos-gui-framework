
import os,subprocess,json,fcntl,time,signal,datetime,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parent
r=Path('/rscratch/wzzheng/experiments/gui-libra-low-terminal-cefm500-a22-20261014-v1/source').resolve()
py='/rscratch/wzzheng/code/cosmos-gui/.local/gui-venv/bin/python'
def free():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 busy={s.split(',')[0].strip() for s in procs.splitlines()}
 available=[]
 for row in rows:
  i,u,n,m,v=[x.strip() for x in row.split(',')]
  if n=='NVIDIA RTX A6000' and int(m)<=100 and int(v)==0 and u not in busy:available.append(u)
 assert len(available)>=7,('admission failed',available)
 return available
with (root/'single-card-debug.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 devices=free();assert free()==devices
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 out=p/('attempt-'+stamp);out.mkdir()
 env={k:v for k,v in os.environ.items() if not k.startswith(('GUI_','WANDB_','AUDIT_')) and k not in ('MOT_CHECKPOINT_PATH','AR_ADAPTER_FILE')}
 env.update(LD_LIBRARY_PATH='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(r),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',GUI_DTENSOR_SAFE_INIT='1',GUI_BACKBONE_PATH='/rscratch/wzzheng/code/cosmos-gui/.local/models/GUI-Libra-8B',WAN_VAE_PATH='/rscratch/wzzheng/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth',BASE_CHECKPOINT_PATH='/rscratch/wzzheng/code/cosmos-gui/.local/models/Cosmos3-Nano-DCP',GUI_TRAIN_MANIFEST=str(p/'fixture.jsonl'),GUI_RUN_NAME='finite-fm-'+stamp,IMAGINAIRE_OUTPUT_ROOT=str(out/'runtime'),GUI_HORIZON='1',GUI_INSTRUCTION_LEVEL='low',GUI_MOT_JOINT='1',GUI_HYBRID_AR='0',GUI_HYBRID_KD_WEIGHT='0',GUI_NORMALIZE_PLAN='0',GUI_AR_ACTIVATION_CHECKPOINT='1',GUI_GROUPED_CE='1',GUI_SUPERVISION_CACHE=str(p/'direct-cache.json'),GUI_DATA_SEED='42',GUI_RESPONSE_STYLE='direct',GUI_FSDP_MASTER_DTYPE='float32',GUI_ADAPTER_EXPORT_ROOT=str(out/'adapters'),GUI_ADAPTER_EXPORT_EVERY='1',PYTORCH_ALLOC_CONF='expandable_segments:True',CUDA_VISIBLE_DEVICES='')
 args=[str(p/'finite_entry_world7.py'),'--sft-toml',str(p/'train-world7.toml')]
 with (out/'dryrun.log').open('w') as f:
  check=subprocess.run([py]+args+['--dryrun'],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=150)
 if check.returncode:
  print('DRYRUN_FAILED',out,flush=True);raise SystemExit(check.returncode)
 assert free()==devices
 env['CUDA_VISIBLE_DEVICES']=','.join(devices[:7])
 cmd=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=7']+args+['--','trainer.seed=42','job.wandb_mode=disabled']
 with (out/'train.log').open('w') as f:
  proc=subprocess.Popen(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  active=dict(pid=proc.pid,command=cmd,uuids=devices[:7],out=str(out),world_size=7,global_batch=7,max_iter=1,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  (p/'active.json').write_text(json.dumps(active,indent=2));print('STARTED',json.dumps(active),flush=True)
  try:code=proc.wait(timeout=900)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=124
 (out/'exit.json').write_text(json.dumps(dict(exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
 print('EXIT',code,out,flush=True)
