import os,json,time,hashlib,concurrent.futures,fcntl,traceback,tarfile,subprocess,signal
from pathlib import Path
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from huggingface_hub import hf_hub_download
root=Path('/data/mai/assets/gui-libra-81k-sft-20261028-v1'); root.mkdir(exist_ok=True)
lock=open(root/'download.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads(Path('/data/mai/assets/download-plan.json').read_text())
(root/'download-plan.json').write_text(json.dumps(plan,indent=2))
state={'stage':'downloading','total_bytes':plan['total_bytes'],'verified':[],'errors':[],'started':time.time()}
def save():
    state['updated']=time.time(); p=root/'download-status.tmp'; p.write_text(json.dumps(state,indent=2)); p.replace(root/'download-status.json')
def fetch(item):
    name=item['path']; print('START',name,flush=True)
    p=root/'snapshot'/name
    p.parent.mkdir(parents=True,exist_ok=True)
    if 'lfs' in item and not p.exists():
        partials=list((root/'snapshot/.cache/huggingface/download'/str(Path(name).parent)).glob('*.'+item['lfs']['oid']+'.incomplete'))
        if len(partials)==1: partials[0].replace(p)
    if not p.exists() or p.stat().st_size!=item['size'] or Path(str(p)+'.aria2').exists():
        cmd=['aria2c','--continue=true','--allow-overwrite=true','--auto-file-renaming=false','--file-allocation=none','--max-connection-per-server=8','--split=8','--min-split-size=16M','--max-tries=3','--retry-wait=5','--timeout=60','--connect-timeout=20','--summary-interval=0','--console-log-level=warn','--dir='+str(p.parent),'--out='+p.name]
        if 'lfs' in item: cmd.append('--checksum=sha-256='+item['lfs']['oid'])
        url='https://hf-mirror.com/datasets/'+plan['dataset']+'/resolve/'+plan['revision']+'/'+name
        failures=0
        for refresh in range(16):
            with (root/(p.name+'.transfer.log')).open('a') as log:
                proc=subprocess.Popen(cmd+[url+'?download=true&refresh='+str(time.time_ns())],stdout=log,stderr=subprocess.STDOUT)
                try: code=proc.wait(timeout=1500)
                except subprocess.TimeoutExpired:
                    proc.send_signal(signal.SIGINT)
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    print('REFRESH_LINK',name,flush=True)
                    continue
            if code==0:break
            failures+=1
            if failures>=3:raise RuntimeError(name+' failed three refreshed download attempts')
            time.sleep(5)
        else:raise RuntimeError(name+' exceeded download refresh limit')
    assert p.stat().st_size==item['size'],name+' size mismatch'
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024): h.update(b)
    digest=h.hexdigest()
    if 'lfs' in item: assert digest==item['lfs']['oid'],name+' SHA256 mismatch'
    print('VERIFIED',name,flush=True)
    return {'path':name,'size':item['size'],'sha256':digest}
save()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures={pool.submit(fetch,item):item for item in plan['files']}
    for fut in concurrent.futures.as_completed(futures):
        try: state['verified'].append(fut.result())
        except Exception as e: state['errors'].append({'path':futures[fut]['path'],'error':str(e)}); print(traceback.format_exc(),flush=True)
        save()
if state['errors']: state['stage']='failed'; save(); raise SystemExit(1)
state['stage']='download_complete';save()
print('ALL_ARCHIVES_VERIFIED',flush=True)



