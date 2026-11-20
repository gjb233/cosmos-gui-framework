from pathlib import Path
import os,signal,time
targets=['/data/mai/start_eval_when_ready.py','/data/mai/run_eval_queue.py','/data/mai/eval_worker.py']
for target in targets:
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().decode().split('\x00')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if target in args:
   try:os.kill(int(p.name),signal.SIGTERM);print('STOPPED_HF_EVAL',p.name,target,flush=True)
   except ProcessLookupError:pass
