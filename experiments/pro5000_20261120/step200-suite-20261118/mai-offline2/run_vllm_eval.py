from pathlib import Path
import subprocess,json,time,os,urllib.request,signal,fcntl
R=Path('/data/mai/evaluation/step200-suite-20261118/mai-offline2');R.mkdir(parents=True,exist_ok=True)
lock=(R/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert Path('/data/mai/training/mai-ui-zero2-mlp-strict10k-20261113/run/pause-request-400.json').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
base='/data/mai/benchmarks/gb256-20261112/assets/mai-base/model'
A=Path('/data/mai/evaluation/assets-20261113')
ground=[json.loads(l) for l in (A/'grounding/manifest.jsonl').open()]
web=[json.loads(l) for l in (A/'mind2web/manifest.jsonl').open()]
sizes=[(r['screenshot_width'],r['screenshot_height']) for r in ground]+[(im['width'],im['height']) for r in web for im in r['images']]
max_pixels=max(((w+31)//32*32)*((h+31)//32*32) for w,h in sizes)
(R/'image-policy.json').write_text(json.dumps(dict(min_pixels=1024,max_pixels=max_pixels,scope='ceiling32 area of largest test image; no test image downscaled by cap')))
assignments=[2]*8;servers=[];clients=[]
env=os.environ.copy();env.update(LD_LIBRARY_PATH='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
env.update(PATH='/data/mai/envs/vllm-eval/bin:/usr/local/cuda/bin:'+env.get('PATH',''),CUDA_HOME='/usr/local/cuda',MAX_JOBS='4',TORCHINDUCTOR_COMPILE_THREADS='4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
try:
 for gpu,step in enumerate(assignments):
  model=base if not step else str(R/'models'/f'step-{step:06d}')
  if step:assert (Path(model)/'merge-verified.json').exists()
  port=25500+gpu
  cmd=['/data/mai/envs/vllm-eval/bin/python','-m','vllm.entrypoints.openai.api_server','--model',model,'--served-model-name','eval','--host','127.0.0.1','--port',str(port),'--dtype','float16','--tensor-parallel-size','1','--max-model-len','32768','--max-num-seqs','64','--max-num-batched-tokens','8192','--gpu-memory-utilization','0.88','--limit-mm-per-prompt','{"image":1,"video":0}','--mm-processor-kwargs',json.dumps(dict(min_pixels=1024,max_pixels=max_pixels)),'--no-enable-prefix-caching']
  e=dict(env,CUDA_VISIBLE_DEVICES=str(gpu))
  p=subprocess.Popen(cmd,env=e,stdin=subprocess.DEVNULL,stdout=(R/f'server-gpu{gpu}.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True);servers.append(p)
  (R/f'server-gpu{gpu}.json').write_text(json.dumps(dict(pid=p.pid,step=step,gpu=gpu,port=port,command=cmd)))
  time.sleep(2)
 deadline=time.time()+1800;ready=set()
 while len(ready)<8:
  assert all(p.poll() is None for p in servers),'vLLM server exited'
  for gpu in range(8):
   if gpu in ready:continue
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{25500+gpu}/v1/models',timeout=2) as response:assert response.status==200
    ready.add(gpu);print('SERVER_READY',gpu,flush=True)
   except Exception:pass
  assert time.time()<deadline,'server startup timeout'
  time.sleep(5)
 ports={step:','.join(str(25500+i) for i,s in enumerate(assignments) if s==step) for step in set(assignments)}
 for step in (2,):
  with (R/f'smoke-{step}.log').open('a') as log:
   subprocess.run(['/data/mai/envs/mai/bin/python','/data/mai/step200-suite-20261118/mai-offline2/vllm_eval_client.py','--step',str(step),'--ports',ports[step],'--concurrency','32','--limit','32'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 (R/'smoke-passed.json').write_text(json.dumps(dict(passed=True,concurrency=32)))
 for step in (2,):
  concurrency=64*ports[step].count(',')+64
  cmd=['/data/mai/envs/mai/bin/python','/data/mai/step200-suite-20261118/mai-offline2/vllm_eval_client.py','--step',str(step),'--ports',ports[step],'--concurrency',str(concurrency)]
  p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=(R/f'client-{step}.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True);clients.append(p)
 (R/'active.json').write_text(json.dumps(dict(server_pids=[p.pid for p in servers],client_pids=[p.pid for p in clients],ports=ports,concurrency_per_server=64)))
 while any(p.poll() is None for p in clients):
  assert all(p.poll() is None for p in servers),'vLLM server exited during evaluation'
  assert all(p.poll() in (None,0) for p in clients),'evaluation client failed'
  time.sleep(10)
 assert all(p.returncode==0 for p in clients)
 subprocess.run(['/data/mai/envs/mai/bin/python','/data/mai/step200-suite-20261118/mai-offline2/collect_vllm_eval.py'],check=True,env=env)
 (R/'complete.json').write_text(json.dumps(dict(completed=True)))
except Exception as e:
 (R/'failed.json').write_text(json.dumps(dict(error=repr(e))));raise
finally:
 for p in clients+servers:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
