from pathlib import Path
import os,subprocess,json,time,fcntl
R=Path('/data/mai/evaluation/ckpts-through-200-20261113');R.mkdir(parents=True,exist_ok=True)
lock=(R/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
training=Path('/data/mai/training/mai-ui-zero2-mlp-strict10k-20261113/run')
assert (training/'pause-request-200.json').exists(),'training not paused at 200 yet'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPUs still occupied'
env=os.environ.copy();env.update(LD_LIBRARY_PATH='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
def launch(step,shard,shards,gpu,limit=0):
 e=dict(env,CUDA_VISIBLE_DEVICES=str(gpu));name=f'step{step}-shard{shard}-'+('smoke' if limit else 'full')
 cmd=['/data/mai/envs/mai/bin/python','/data/mai/eval_worker.py','--step',str(step),'--shard',str(shard),'--shards',str(shards)]
 if limit:cmd+=['--limit',str(limit)]
 return subprocess.Popen(cmd,env=e,stdin=subprocess.DEVNULL,stdout=(R/(name+'.log')).open('a'),stderr=subprocess.STDOUT,start_new_session=True)
def wait(procs):
 while any(p.poll() is None for p in procs):
  failed=[p for p in procs if p.poll() not in (None,0)]
  if failed:
   for p in procs:
    if p.poll() is None:p.terminate()
   raise RuntimeError('evaluation worker failed: '+str([(p.pid,p.returncode) for p in failed]))
  time.sleep(5)
 assert all(p.returncode==0 for p in procs)
try:
 smoke=[launch(0,0,4,0,2),launch(200,0,4,1,2)];wait(smoke)
 (R/'smoke-passed.json').write_text(json.dumps(dict(passed=True)))
 for steps in ((0,200),(100,150),(50,)):
  shards=4 if len(steps)==2 else 8
  procs=[launch(step,shard,shards,j*shards+shard) for j,step in enumerate(steps) for shard in range(shards)]
  (R/'active.json').write_text(json.dumps(dict(steps=steps,worker_pids=[p.pid for p in procs],started=time.time())))
  wait(procs)
  subprocess.run(['/data/mai/envs/mai/bin/python','/data/mai/collect_eval.py'],check=True,env=env)
 (R/'complete.json').write_text(json.dumps(dict(completed=True)))
except Exception as e:
 (R/'failed.json').write_text(json.dumps(dict(error=repr(e))));raise
