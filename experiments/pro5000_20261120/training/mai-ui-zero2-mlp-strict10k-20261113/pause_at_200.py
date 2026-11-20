from pathlib import Path
import json,time,os,signal
W=Path('/data/mai/training/mai-ui-zero2-mlp-strict10k-20261113');R=W/'run'
active=json.loads((R/'active.json').read_text());pid=active['pid']
while Path(f'/proc/{pid}').exists():
 p=R/'checkpoints/latest-complete.json'
 if p.exists():
  state=json.loads(p.read_text())
  if state['completed_steps']>=200:
   ck=R/'checkpoints'/state['tag']
   assert len(list(ck.glob('*optim_states.pt')))==8
   assert (R/'ar-adapters/ar-step-000200.ready.json').exists()
   cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' ')
   assert str(W/'train_entry.py') in cmd and os.getpgid(pid)==pid
   os.killpg(pid,signal.SIGTERM)
   (R/'pause-request-200.json').write_text(json.dumps(dict(reason='user_requested_pause_at_step_200',checkpoint=state['tag'],pid=pid,signal_sent_at=time.time())))
   print('PAUSED_AFTER_CHECKPOINT',state['tag'],flush=True)
   break
 time.sleep(2)
else:print('TRAINING_ALREADY_EXITED',flush=True)
