import os,time,subprocess,fcntl,json
from pathlib import Path
import torch
d=Path(__file__).resolve().parent
uid=os.environ['CUDA_VISIBLE_DEVICES']
idx=os.environ['HOLD_CARD']
lock=(d/f'gpu{idx}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
    info=subprocess.check_output(['nvidia-smi','-i',uid,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if int(info)>=48000 and not subprocess.check_output(['nvidia-smi','-i',uid,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        try:
            n=46592*1024**2
            buf=torch.empty(n,device='cuda',dtype=torch.uint8)
            torch.cuda.synchronize()
            (d/f'gpu{idx}.held.json').write_text(json.dumps(dict(pid=os.getpid(),uuid=uid,allocated_MiB=buf.numel()//1024**2)))
            while True:time.sleep(30)
        except torch.OutOfMemoryError:
            raise
    time.sleep(4)
