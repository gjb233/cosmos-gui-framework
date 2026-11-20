from pathlib import Path
import os,json,time,subprocess,signal,fcntl,urllib.request
V=Path(__file__).resolve().parent;E=V
lock=(V/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
u='GPU-5332ba0e-bad7-4f31-deb4-b7f51ecaff90'
gl=Path('/tmp/mixed-eight-eval-'+u+'.lock').open('a');fcntl.flock(gl,fcntl.LOCK_EX|fcntl.LOCK_NB)
def free():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 return int(next(r.split(',')[1] for r in rows.splitlines() if r.startswith(u)))
gl5=Path('/tmp/mixed-eight-eval-GPU-ce3c4442-a6a8-4975-1119-9aa4aeaa619f.lock').open('a');fcntl.flock(gl5,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader'],text=True).splitlines()
assert rows[0].split(',')[1].strip()=='GPU-ce3c4442-a6a8-4975-1119-9aa4aeaa619f'
assert rows[5].split(',')[1].strip()==u
assert [r.split(',')[2].strip() for r in rows]==sorted(r.split(',')[2].strip() for r in rows)
PY='/rscratch/wzzheng/experiments/gui-mai-official-alignment-20261006-v1/official-venv/bin/python'
assert subprocess.check_output(['hostname'],text=True).strip()=='a23'
cfg=json.loads((E/'grounding/config.json').read_text())
cmd=[PY,'-m','vllm.entrypoints.openai.api_server','--model',cfg['model'],'--served-model-name','base','--host','127.0.0.1','--port','25451','--dtype','float16','--tensor-parallel-size','2','--max-model-len','32768','--max-num-seqs','4','--gpu-memory-utilization','.42','--limit-mm-per-prompt','{"image":1,"video":0}','--enforce-eager','--no-enable-prefix-caching','--cpu-offload-gb','0','--enable-lora','--max-lora-rank','16','--max-loras','1','--lora-modules','eval='+str(V/'adapter'),'--max-num-batched-tokens','2048']
env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,5',PYTHONPATH=str(V/'compat'),MAI_FIX_MM_LORA='1',LD_LIBRARY_PATH='',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
with (V/'server.log').open('a') as f:server=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(V/'active.json').write_text(json.dumps(dict(server_pid=server.pid,controller_pid=os.getpid(),uuid=u,command=cmd,started=time.time()),indent=2))
code=1
try:
 deadline=time.time()+1800
 while True:
  assert server.poll() is None,'server exited; see server.log'
  try:
   with urllib.request.urlopen('http://127.0.0.1:25451/v1/models',timeout=5) as r:assert r.status==200
   break
  except Exception:
   assert time.time()<deadline,'server startup timed out'
   time.sleep(5)
 for name in os.environ.get('MAI_EVAL_PHASES','grounding,mind2web').split(','):
  d=V/name
  with (d/'client.log').open('a') as f:code=subprocess.call([PY,str(d/'client.py')],env=env,cwd=str(d),stdout=f,stderr=subprocess.STDOUT)
  assert code==0,name+' failed'
 (V/'complete.json').write_text(json.dumps(dict(completed=True,results={n:json.loads((V/n/'results.json').read_text()) for n in os.environ.get('MAI_EVAL_PHASES','grounding,mind2web').split(',')})))
finally:
 if server.poll() is None:
  os.killpg(server.pid,signal.SIGTERM)
  try:server.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
 (V/'exit.json').write_text(json.dumps(dict(exit_code=code,finished=time.time())))
