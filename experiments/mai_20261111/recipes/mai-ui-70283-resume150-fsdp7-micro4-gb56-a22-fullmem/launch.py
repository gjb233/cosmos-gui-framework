from pathlib import Path
import os,json,time,subprocess,fcntl,signal,datetime
S=Path(__file__).resolve().parent;P=S.parent
PY='/rscratch/wzzheng/code/cosmos-gui/.local/gui-venv/bin/python'
SOURCE='/rscratch/wzzheng/experiments/gui-grounding-fixed-ab-20261002-v1/source'
assert subprocess.check_output(['hostname'],text=True).strip()=='a22'
lock=(S/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
uuids=['GPU-8d145fbd-3881-c721-56ae-4ff8c7496431', 'GPU-cecb7b97-8b95-682b-abdd-5f49b1384b03', 'GPU-069a7bee-404b-9538-cdef-1e0173979b86', 'GPU-1ac71373-3c0c-b06b-0d98-f4200602ddd4', 'GPU-857fe24c-f9c1-2db3-846f-ecc8430a1fe3', 'GPU-2ea1ff61-df07-3fbd-a934-27d20f018c79', 'GPU-3920c117-8036-b3e5-c092-2eb998f5b89a']
uuid=','.join(uuids)
gpu_locks=[]
for uid in uuids:
 gl=Path('/tmp/mixed-eight-eval-'+uid+'.lock').open('a');fcntl.flock(gl,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_locks.append(gl)
def status(stage,**kw):(S/'status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2))
def gpu():
 data=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
 for uid in uuids:
  row=next(x.split(',') for x in data if uid in x);assert int(row[2])>=22000,data
def train(validation):
 gpu()
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 out=S/('validation-runs' if validation else 'runs')/stamp;out.mkdir(parents=True)
 env={k:v for k,v in os.environ.items() if not k.startswith(('GUI_','WANDB_','AUDIT_')) and k not in ('MOT_CHECKPOINT_PATH','AR_ADAPTER_FILE','GUI_TARGET_HEIGHT','GUI_TARGET_WIDTH')}
 env.update(LD_LIBRARY_PATH='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=SOURCE,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',GUI_SUPERVISION_CACHE=str(P/'fm-debug-draft/direct-cache.json'),PYTORCH_ALLOC_CONF='expandable_segments:True',
 GUI_ADAPTER_EXPORT_ROOT=str(out/'adapters'),GUI_ADAPTER_EXPORT_EVERY='1' if validation else '25',GUI_RUN_NAME='mai-native-allsource-'+stamp,IMAGINAIRE_OUTPUT_ROOT=str(out/'runtime'))
 args=[str(S/'train_entry.py'),'--config',str(S/('validation-config.json' if validation else 'config.json')),'--manifest',str(S/'artifacts/train-manifest.jsonl'),'--phase1',str(S/'artifacts'/('validation-phase1.jsonl' if validation else 'schedule-phase1.jsonl')),'--phase2',str(S/'artifacts'/('validation-phase2.jsonl' if validation else 'schedule-phase2.jsonl')),'--native-config',str(S/('validation-native.toml' if validation else 'native.toml'))]
 with (out/'dryrun.log').open('w') as f:subprocess.run([PY]+args+['--dryrun'],env=env,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,check=True)
 gpu()
 cmd=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=7']+args
 with (out/'train.log').open('w') as f:
  proc=subprocess.Popen(cmd,env=env,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  a=dict(pid=proc.pid,command=cmd,out=str(out),uuid=uuid,host='a22',validation=validation,model='MAI-UI-8B',fresh=False,resume_from=150,world_size=7,global_batch=14 if validation else 56)
  (S/('validation-active.json' if validation else 'active.json')).write_text(json.dumps(a,indent=2));status('validating' if validation else 'training',**a)
  if not validation:
   W=S/'wandb-sync';W.mkdir(exist_ok=True)
   wp=subprocess.Popen([PY,str(W/'sync.py')],env=env,cwd=str(W),stdout=(W/'sync.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
   (W/'active.json').write_text(json.dumps(dict(pid=wp.pid,command=[PY,str(W/'sync.py')],host='a22')))
  code=proc.wait()
 (out/'exit.json').write_text(json.dumps(dict(exit_code=code,time=time.time())))
 assert code==0,('training failed',out,code)
 if validation:
  for rank in range(7):
   done=[json.loads(l) for l in (out/f'adapters/complete-rank{rank}.jsonl').open()][-1]
   assert done['passed'] and done['changed']['ar']>0 and done['changed']['fm']>0
  (S/'validation-passed.json').write_text(json.dumps(dict(passed=True,out=str(out),sources=8)))
 return out
train(False)
status('complete')
