"""One task lock, seven idle GPU UUIDs, audited manifest, native dryrun."""
import os,json,subprocess,fcntl,datetime,math
from pathlib import Path
P=Path(__file__).resolve().parent
PY='/rscratch/wzzheng/code/cosmos-gui/.local/gui-venv/bin/python'
def free():
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 busy={x.split(',')[0].strip() for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()}
 result=[]
 for line in gpu.splitlines():
  u,n,m,v=[x.strip() for x in line.split(',')]
  if n=='NVIDIA RTX A6000' and int(m)<=100 and int(v)==0 and u not in busy:result.append(u)
 if len(result)<7:raise RuntimeError('need seven simultaneously idle A6000 GPUs')
 return result[:7]
def main():
 from training_backend import validate_inputs
 with (P/'single-card-debug.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  subprocess.run([PY,str(P/'preflight.py')],cwd=P,check=True)
  cfg,records,draws=validate_inputs(P/'mixture.json',P/'artifacts/train-manifest.jsonl',[P/'artifacts/schedule-phase1.jsonl',P/'artifacts/schedule-phase2.jsonl'])
  devices=free();assert free()==devices
  stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  out=P/'training-runs'/stamp;out.mkdir(parents=True,exist_ok=False)
  steps=math.ceil(len(draws)/cfg['train']['global_batch'])
  warmup=max(1,int(steps*cfg['train']['warmup_fraction']))
  toml=(P/'unified-template-debug/train-world7.toml').read_text()
  toml=toml.replace('max_iter = 1','max_iter = '+str(steps)).replace('grad_accum_iter = 1','grad_accum_iter = 16')
  toml=toml.replace('save_iter = 1000000','save_iter = 100')
  toml=toml.replace('cycle_lengths = [1]','cycle_lengths = ['+str(steps)+']').replace('warm_up_steps = [0]','warm_up_steps = ['+str(warmup)+']')
  toml=toml.replace('f_start = [1.0]','f_start = [0.0]').replace('f_min = [1.0]','f_min = [0.0]')
  native=out/'native.toml';native.write_text(toml)
  env=dict(os.environ,LD_LIBRARY_PATH='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
   PYTHONPATH='/rscratch/wzzheng/experiments/gui-grounding-fixed-ab-20261002-v1/source',
   CUDA_VISIBLE_DEVICES='',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',
   GUI_SUPERVISION_CACHE=str(P/'fm-debug-draft/direct-cache.json'),
   GUI_ADAPTER_EXPORT_ROOT=str(out/'adapters'),GUI_ADAPTER_EXPORT_EVERY='100',
   GUI_RUN_NAME='mixed-'+stamp,IMAGINAIRE_OUTPUT_ROOT=str(out/'runtime'),
   PYTORCH_ALLOC_CONF='expandable_segments:True')
  args=[str(P/'training_backend.py'),'--config',str(P/'mixture.json'),'--manifest',str(P/'artifacts/train-manifest.jsonl'),
        '--phase1',str(P/'artifacts/schedule-phase1.jsonl'),'--phase2',str(P/'artifacts/schedule-phase2.jsonl'),'--native-config',str(native)]
  with (out/'dryrun.log').open('w') as log:
   subprocess.run([PY]+args+['--dryrun'],env=env,cwd='/rscratch/wzzheng/experiments/gui-grounding-fixed-ab-20261002-v1/source',stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
  assert free()==devices
  env['CUDA_VISIBLE_DEVICES']=','.join(devices)
  cmd=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=7']+args
  with (out/'train.log').open('w') as log:
   child=subprocess.Popen(cmd,env=env,cwd='/rscratch/wzzheng/experiments/gui-grounding-fixed-ab-20261002-v1/source',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   (P/'training-active.json').write_text(json.dumps(dict(pid=child.pid,command=cmd,uuids=devices,output=str(out),steps=steps,world_size=7,global_batch=112,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
   code=child.wait()
  (out/'exit.json').write_text(json.dumps(dict(exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
  raise SystemExit(code)
if __name__=='__main__':main()
