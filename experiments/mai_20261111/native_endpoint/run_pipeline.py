from pathlib import Path
import os,json,time,subprocess,fcntl,signal,datetime
S=Path(__file__).resolve().parent;P=S.parent
PY='/rscratch/wzzheng/code/cosmos-gui/.local/gui-venv/bin/python'
SOURCE='/rscratch/wzzheng/experiments/gui-grounding-fixed-ab-20261002-v1/source'
assert subprocess.check_output(['hostname'],text=True).strip()=='a18'
lock=(S/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
uuids=['GPU-59e5f650-c035-75f5-987e-563bbbdf8762', 'GPU-6eaa65a5-fd29-10ab-d2a1-362103379636', 'GPU-964ccee1-7a00-fe7e-1fac-8fa8fdc95b5e', 'GPU-43e90bd7-c2a1-1754-1eba-379d4a6890f3']
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
 env.update(LD_LIBRARY_PATH='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=SOURCE,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',GUI_ACTIVATION_CPU_OVERFLOW='1',GUI_SUPERVISION_CACHE=str(P/'fm-debug-draft/direct-cache.json'),PYTORCH_ALLOC_CONF='expandable_segments:True',
 GUI_ADAPTER_EXPORT_ROOT=str(out/'adapters'),GUI_ADAPTER_EXPORT_EVERY='1' if validation else '50',GUI_RUN_NAME='mai-native-allsource-'+stamp,IMAGINAIRE_OUTPUT_ROOT=str(out/'runtime'))
 args=[str(S/'train_entry.py'),'--config',str(S/('validation-config.json' if validation else 'config.json')),'--manifest',str(S/'artifacts/train-manifest.jsonl'),'--phase1',str(S/'artifacts'/('validation-phase1.jsonl' if validation else 'schedule-phase1.jsonl')),'--phase2',str(S/'artifacts'/('validation-phase2.jsonl' if validation else 'schedule-phase2.jsonl')),'--native-config',str(S/('validation-native.toml' if validation else 'native.toml'))]
 with (out/'dryrun.log').open('w') as f:subprocess.run([PY]+args+['--dryrun'],env=env,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,check=True)
 gpu()
 cmd=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=4']+args
 with (out/'train.log').open('w') as f:
  proc=subprocess.Popen(cmd,env=env,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  a=dict(pid=proc.pid,command=cmd,out=str(out),uuid=uuid,host='a18',validation=validation,model='MAI-UI-8B',fresh=True,world_size=4,global_batch=8 if validation else 112)
  (S/('validation-active.json' if validation else 'active.json')).write_text(json.dumps(a,indent=2));status('validating' if validation else 'training',**a)
  if not validation:
   W=S/'wandb-sync';W.mkdir(exist_ok=True)
   wp=subprocess.Popen([PY,str(W/'sync.py')],env=env,cwd=str(W),stdout=(W/'sync.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
   (W/'active.json').write_text(json.dumps(dict(pid=wp.pid,command=[PY,str(W/'sync.py')],host='a18')))
  code=proc.wait()
 (out/'exit.json').write_text(json.dumps(dict(exit_code=code,time=time.time())))
 assert code==0,('training failed',out,code)
 if validation:
  for rank in range(4):
   done=[json.loads(l) for l in (out/f'adapters/complete-rank{rank}.jsonl').open()][-1]
   assert done['passed'] and done['changed']['ar']>0 and done['changed']['fm']>0
  (S/'validation-passed.json').write_text(json.dumps(dict(passed=True,out=str(out),sources=8)))
 return out
try:
 assert json.loads((S/'data-ready.json').read_text())['passed']
 assert json.loads((S/'token-audit.json').read_text())['passed']
 status('waiting_for_data_build')
 while not (S/'build-report.json').exists():
  assert Path('/proc/970655').exists(),'data builder exited before completion'
  time.sleep(10)
 report=json.loads((S/'build-report.json').read_text());assert all(x['fm']>0 for x in report['counts'].values()) and report['sources']==8
 rows=[json.loads(l) for l in (S/'artifacts/train-manifest.jsonl').open()]
 selected=[]
 for src in sorted(report['counts']):
  rr=[r for r in rows if r['source']==src and r['fm_mask']]
  # Moderate size fixture per source; all full-sized samples remain in formal training.
  r=min(rr,key=lambda x:abs(x['transition_provenance']['current_size'][0]*x['transition_provenance']['current_size'][1]-1000000))
  selected.append(dict(draw=len(selected),id=r['id'],source=src,level='high',variant='direct',fm_mask=1,padding=False))
 selected=selected[:8]
 selected=[dict(x,draw=i) for i,x in enumerate(selected)]
 (S/'artifacts/validation-phase1.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected))
 (S/'artifacts/validation-phase2.jsonl').write_text('')
 cfg=json.loads((S/'config.json').read_text());cfg['train'].update(gradient_accumulation=2,global_batch=8,max_iter=1)
 (S/'validation-config.json').write_text(json.dumps(cfg,indent=2))
 txt=(S/'native.toml').read_text().replace('max_iter = 2024','max_iter = 1').replace('grad_accum_iter = 28','grad_accum_iter = 2').replace('save_iter = 50','save_iter = 100').replace('cycle_lengths = [2024]','cycle_lengths = [1]').replace('warm_up_steps = [20]','warm_up_steps = [0]')
 (S/'validation-native.toml').write_text(txt)
 train(True)
 train(False)
 status('complete')
except Exception as e:
 status('failed',error=repr(e));raise
