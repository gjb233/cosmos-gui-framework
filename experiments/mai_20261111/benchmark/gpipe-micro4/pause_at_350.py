from pathlib import Path
import os,json,time,signal,fcntl
import torch
from safetensors.torch import load_file
from torch.distributed.checkpoint import FileSystemReader
O=Path(__file__).resolve().parent;S=O.parent/'mai-ui-70283-resume150-fsdp7-micro4-gb56-a22-fullmem'
lock=(O/'pause.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
a=json.loads((S/'active.json').read_text());out=Path(a['out']);pid=a['pid']
assert pid==4127123
ck=out/'runtime/cosmos-gui/androidcontrol/mai-native-allsource-20261110T090029Z/checkpoints/iter_000000350'
ready=out/'adapters/ar-step-000350.ready.json'
while not ready.exists() or not all((ck/k/'.metadata').exists() for k in ['model','optim','scheduler','trainer']):
 assert Path(f'/proc/{pid}').exists(),'old training exited before 350'
 time.sleep(5)
audit={}
for k in ['model','optim','scheduler','trainer']:
 md=FileSystemReader(str(ck/k)).read_metadata()
 for entry in md.storage_data.values():
  f=ck/k/entry.relative_path
  assert f.stat().st_size>=entry.offset+entry.length,(k,f)
 audit[k]=dict(metadata_keys=len(md.state_dict_metadata),shards=[dict(name=f.name,bytes=f.stat().st_size) for f in sorted((ck/k).glob('*.distcp'))])
 assert len(audit[k]['shards'])==7
w=load_file(str(out/'adapters/ar-step-000350.safetensors'))
assert len(w)==288 and all(torch.isfinite(v).all() for v in w.values())
for p in [pid]:
 cmd=Path(f'/proc/{p}/cmdline').read_bytes().decode().replace('\0',' ')
 assert str(S/'train_entry.py') in cmd,cmd
 # torchrun owns a session, terminate only its process group.
 assert os.getpgid(p)==p
 os.killpg(p,signal.SIGTERM)
for _ in range(60):
 if not Path(f'/proc/{pid}').exists():break
 time.sleep(1)
else:raise RuntimeError('old process did not exit; inspect manually')
result=dict(step=350,next_draw=9600+(350-150)*56,checkpoint=str(ck),stopped_pid=pid,adapter_finite=288,metadata=audit,requested='switch to 7-stage 1F1B micro4 global56')
(O/'paused-350.json').write_text(json.dumps(result,indent=2))
print('PAUSED_AT_350',json.dumps(result),flush=True)
