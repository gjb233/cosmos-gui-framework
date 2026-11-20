from pathlib import Path
import subprocess,time,os,json,signal,urllib.request,concurrent.futures,fcntl
R=Path('/data/mai/evaluation/step200-suite-20261118/mai-aw200');R.mkdir(parents=True,exist_ok=True);B=Path('/data/mai/benchmarks/androidworld');sdk=B/'sdk'
lock=(R/'parallel.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(slot,phase,**data):
 p=R/f'worker-{slot}.tmp';p.write_text(json.dumps(dict(slot=slot,phase=phase,time=time.time(),**data)));p.replace(R/f'worker-{slot}.json')
def worker(slot,steps):
 emu=None;server=None
 try:
  env=os.environ.copy();tmp=B/f'worker-tmp-v2/{slot}';(tmp/'android_world').mkdir(parents=True,exist_ok=True)
  cache=tmp/'android_world/app_data'
  if not cache.exists():cache.symlink_to('/tmp/android_world/app_data',target_is_directory=True)
  env.update(TMPDIR=str(tmp),ANDROID_AVD_HOME=str(B/'parallel-avd-v2'),ANDROID_HOME=str(sdk),CUDA_VISIBLE_DEVICES=str(slot),PATH='/data/mai/envs/vllm-eval/bin:/usr/local/cuda/bin:'+env['PATH'],CUDA_HOME='/usr/local/cuda',LD_LIBRARY_PATH='',OMP_NUM_THREADS='1',MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1')
  console=5554+slot*2;grpc=8554+slot;port=25600+slot
  env['ANDROID_ADB_SERVER_PORT']=str(5040+slot);env['ADB_SERVER_SOCKET']=f'tcp:127.0.0.1:{5040+slot}'
  adb=f'/data/mai/step200-suite-20261118/mai-aw200/adb-{slot}'
  subprocess.run([adb,'start-server'],env=env,check=True)
  status(slot,'booting_emulator')
  emu=subprocess.Popen([str(sdk/'emulator/emulator'),'-avd',f'AndroidWorldSlot{slot}','-port',str(console),'-grpc',str(grpc),'-no-window','-no-snapshot','-no-audio','-no-boot-anim','-gpu','swiftshader','-memory','4096'],env=env,stdout=(R/f'emulator-{slot}.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
  deadline=time.time()+300
  while True:
   assert emu.poll() is None,'emulator exited'
   result=subprocess.run([adb,'-s',f'emulator-{console}','shell','getprop','sys.boot_completed'],capture_output=True,text=True,timeout=15)
   if result.stdout.strip()=='1':break
   assert time.time()<deadline,'emulator boot timeout'
   time.sleep(3)
  env['AW_REBUILD_SNAPSHOTS']='1'
  for step in steps:
   if (R/'full'/f'step-{step:06d}'/f'slot-{slot}'/'complete.json').exists():continue
   model='/data/mai/evaluation/vllm-ckpts-through-200-20261113/models/step-000200'
   status(slot,'starting_server',step=step)
   cmd=['/data/mai/envs/vllm-eval/bin/python','-m','vllm.entrypoints.openai.api_server','--model',model,'--served-model-name','eval','--host','127.0.0.1','--port',str(port),'--dtype','float16','--max-model-len','32768','--max-num-seqs','16','--max-num-batched-tokens','8192','--gpu-memory-utilization','0.88','--limit-mm-per-prompt','{"image":1,"video":0}','--mm-processor-kwargs','{"min_pixels":1024,"max_pixels":3000000}','--no-enable-prefix-caching']
   server=subprocess.Popen(cmd,env=env,stdout=(R/f'parallel-server-{step}-slot{slot}.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
   deadline=time.time()+1800
   while True:
    assert server.poll() is None,'server exited'
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as resp:
      if resp.status==200:break
    except Exception:pass
    assert time.time()<deadline,'server timeout'
    time.sleep(5)
   status(slot,'evaluating',step=step)
   with (R/f'eval-{step}-slot{slot}.log').open('a') as f:
    subprocess.run(['/data/mai/envs/androidworld/bin/python','-u','/data/mai/step200-suite-20261118/mai-aw200/androidworld_eval.py','--step',str(step),'--port',str(port),'--console-port',str(console),'--grpc-port',str(grpc),'--slot',str(slot),'--slots','8'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   os.killpg(server.pid,signal.SIGTERM);server.wait(timeout=120);server=None
  status(slot,'completed',steps=steps)
 except Exception as e:
  status(slot,'failed',error=repr(e));raise
 finally:
  for p in (server,emu):
   if p and p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    p.wait(timeout=120)
assignments=[[200] for _ in range(8)]
(R/'parallel-plan.json').write_text(json.dumps(dict(assignments=assignments,seed=30,slots=8,resume_existing_episodes=True)))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 futures=[pool.submit(worker,i,s) for i,s in enumerate(assignments)]
 for f in concurrent.futures.as_completed(futures):f.result()
subprocess.run(['/data/mai/envs/androidworld/bin/python','/data/mai/step200-suite-20261118/mai-aw200/collect_androidworld.py'],check=True)
(R/'parallel-complete.json').write_text(json.dumps(dict(completed=True)))
