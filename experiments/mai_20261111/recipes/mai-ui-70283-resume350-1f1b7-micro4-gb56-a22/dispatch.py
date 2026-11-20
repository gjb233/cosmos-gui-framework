from pathlib import Path
import os,json,time,subprocess,fcntl,signal
O=Path(__file__).resolve().parent;P=O.parent
R=P/'mai-framework-comparison-a22-seven-20261107'
PY='/rscratch/wzzheng/code/cosmos-gui/.local/gui-venv/bin/python'
SOURCE='/rscratch/wzzheng/experiments/gui-grounding-fixed-ab-20261002-v1/source'
assert os.uname().nodename=='a22'
lock=(O/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(stage,**kw):(O/'status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2))
state('waiting_fsdp_baseline')
while True:
 f=O/'fsdp-baseline/status.json'
 if f.exists():
  s=json.loads(f.read_text())
  if s['stage']=='complete':break
  active=O/'fsdp-baseline/active.json'
  if active.exists():
   a=json.loads(active.read_text());ex=Path(a['out'])/'exit.json'
   if ex.exists() and json.loads(ex.read_text())['exit_code']!=0:
    raise RuntimeError('isolated FSDP baseline failed; inspect before launch')
 time.sleep(5)
while not (O/'cpu-stage-audit.json').exists():time.sleep(5)
assert json.loads((O/'cpu-stage-audit.json').read_text())['passed']
old=json.loads((P/'mai-ui-70283-resume150-fsdp7-micro4-gb56-a22-fullmem/active.json').read_text())
uuids=old['uuid'].split(',');locks=[]
for uid in uuids:
 f=open('/tmp/mixed-eight-eval-'+uid+'.lock','a')
 while True:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:state('waiting_gpu_lock',uuid=uid);time.sleep(2)
 locks.append(f)
while True:
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 actual={int(x.split(',')[0]):[v.strip() for v in x.split(',')[1:]] for x in rows.splitlines()}
 assert [actual[i][0] for i in range(7)]==uuids
 if all(int(actual[i][1])>=44000 for i in range(7)):break
 state('waiting_free_memory',snapshot=rows);time.sleep(10)
(O/'preflight.json').write_text(json.dumps(dict(gpus=rows,apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True))))
env={k:v for k,v in os.environ.items() if not k.startswith(('GUI_','WANDB_'))}
env.update(LD_LIBRARY_PATH='',CUDA_VISIBLE_DEVICES=','.join(uuids),CUDA_DEVICE_ORDER='PCI_BUS_ID',
 PYTHONPATH=os.pathsep.join(map(str,[O,R/'deepspeed-env/packages',R/'native-endpoint-reference',P/'unified-template-debug',R/'torch-pipeline',SOURCE])),
 PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',
 GUI_ADAPTER_EXPORT_ROOT=str(O/'producer-audit'),GUI_ADAPTER_EXPORT_EVERY='25',GUI_RUN_NAME='a22-seven-1f1b-resume350',
 IMAGINAIRE_OUTPUT_ROOT=str(O/'producer-runtime'),GUI_SUPERVISION_CACHE=str(P/'fm-debug-draft/direct-cache.json'),
 PYTORCH_ALLOC_CONF='expandable_segments:True',CUBLAS_WORKSPACE_CONFIG=':4096:8',GUI_ACTIVATION_CPU_OVERFLOW='0')
def spawn(cmd,log):
 with log.open('x') as f:return subprocess.Popen(cmd,env=env,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
producer=spawn([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(O/'producer.py'),'--config',str(O/'config.json'),'--manifest',str(O/'artifacts/train-manifest.jsonl'),'--phase1',str(O/'artifacts/schedule-phase1.jsonl'),'--phase2',str(O/'artifacts/schedule-phase2.jsonl'),'--native-config',str(O/'native.toml')],O/'producer.log')
engine=spawn([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=7',str(O/'driver.py')],O/'engine.log')
(O/'active.json').write_text(json.dumps(dict(producer_pid=producer.pid,engine_pid=engine.pid,dispatcher_pid=os.getpid(),gpus=list(range(7)),uuids=uuids,start_step=350,next_draw=20800)))
state('running',producer_pid=producer.pid,engine_pid=engine.pid)
while engine.poll() is None:
 if producer.poll() not in (None,0):
  (O/'stream/abort').touch();os.killpg(engine.pid,signal.SIGTERM);break
 time.sleep(3)
rc=engine.wait()
if rc and producer.poll() is None:
 (O/'stream/abort').touch();os.killpg(producer.pid,signal.SIGTERM)
pc=producer.wait()
(O/'exit.json').write_text(json.dumps(dict(engine=rc,producer=pc)))
state('complete' if rc==pc==0 else 'failed',engine=rc,producer=pc)
