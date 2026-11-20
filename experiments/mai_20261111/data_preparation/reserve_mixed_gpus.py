"""User-authorized, bounded GPU memory reservation; not a training process."""
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/rscratch/wzzheng/experiments/mixed-sft-preparation-20261028-v1')
OUT = ROOT/'reservation-20261101'
BYTES = 40_000_000_000  # exactly 40 decimal GB per GPU, excluding CUDA context
UUIDS = ['GPU-8d145fbd-3881-c721-56ae-4ff8c7496431',
         'GPU-cecb7b97-8b95-682b-abdd-5f49b1384b03',
         'GPU-069a7bee-404b-9538-cdef-1e0173979b86',
         'GPU-1ac71373-3c0c-b06b-0d98-f4200602ddd4',
         'GPU-857fe24c-f9c1-2db3-846f-ecc8430a1fe3',
         'GPU-2ea1ff61-df07-3fbd-a934-27d20f018c79',
         'GPU-3920c117-8036-b3e5-c092-2eb998f5b89a']

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(name, data):
    tmp = OUT/(name+'.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(OUT/name)

def idle():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    gpus = {r[0].strip(): [v.strip() for v in r[1:]] for r in csv.reader(raw.splitlines())}
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader'], text=True)
    occupied = {r[0].strip() for r in csv.reader(processes.splitlines()) if r}
    for uuid in UUIDS:
        name, memory, util = gpus[uuid]
        assert name == 'NVIDIA RTX A6000' and int(memory) <= 100 and int(util) == 0 and uuid not in occupied, uuid

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'active.json').exists():
        raise RuntimeError('Reservation record already exists; inspect before any new attempt')
    with (ROOT/'single-card-debug.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle()
        time.sleep(1)
        idle()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(UUIDS)
        import torch
        assert torch.cuda.device_count() == 7
        idle()  # final check immediately before allocating
        stop = [False]
        def request_stop(*_):
            stop[0] = True
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        buffers = []
        started = now()
        deadline = time.time()+3600
        state = dict(pid=os.getpid(), host=os.uname().nodename, gpu_uuids=UUIDS,
                     bytes_per_gpu=BYTES, started_utc=started, deadline_epoch=deadline,
                     purpose='explicit user-authorized memory reservation, not training')
        try:
            for device in range(7):
                buffers.append(torch.empty(BYTES, dtype=torch.uint8, device=f'cuda:{device}'))
                buffers[-1].zero_()
                torch.cuda.synchronize(device)
            state.update(status='holding', allocated_bytes=[torch.cuda.memory_allocated(i) for i in range(7)])
            write('active.json', state)
            print(json.dumps(state), flush=True)
            while not stop[0] and time.time() < deadline and not (OUT/'release.request').exists():
                time.sleep(2)
            reason = 'requested' if stop[0] or (OUT/'release.request').exists() else 'one_hour_timeout'
        except BaseException as exc:
            reason = repr(exc)
            write('error.json', dict(error=reason, time=now(), pid=os.getpid()))
            raise
        finally:
            buffers.clear()
            for device in range(7):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            write('exit.json', dict(finished_utc=now(), reason=reason, pid=os.getpid()))

if __name__ == '__main__':
    main()
