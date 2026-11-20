"""Read official local TFRecords once; retain selected raw records and provenance.
CPU only. Four shard workers, per-shard atomic results, lock, resume completed shards.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_NUM_INTRAOP_THREADS']='1'
os.environ['TF_NUM_INTEROP_THREADS']='1'
import sys,gzip,struct,json,pathlib,hashlib,concurrent.futures,fcntl,time
sys.path.insert(0,'/rscratch/wzzheng/data/androidcontrol/python')
from tensorflow.core.example.example_pb2 import Example
P=pathlib.Path(__file__).resolve().parent
RAW=pathlib.Path('/rscratch/wzzheng/data/androidcontrol/raw')
OUT=P/'originals/androidcontrol-selected';OUT.mkdir(parents=True,exist_ok=True)
wanted={}
for line in (P/'artifacts/inventory.jsonl').open():
 r=json.loads(line)
 if r['source']=='android_control':
  ep,filename=r['image'].split('/');step=int(filename.removeprefix('screenshot_').removesuffix('.png'))
  wanted.setdefault(int(ep),{})[step]=r['id']
split=json.loads((RAW/'splits.json').read_text())
splitmap={int(ep):key for key in ('train','test','validation') for ep in split[key]}
def atom(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False));t.replace(p)
def worker(shard):
 dest=OUT/(shard.name+'.jsonl');done=OUT/(shard.name+'.complete.json')
 st=shard.stat()
 if done.exists():
  prev=json.loads(done.read_text())
  if prev['size']==st.st_size and prev['mtime_ns']==st.st_mtime_ns and dest.exists():return prev
 temp=dest.with_suffix('.partial');count=0;episodes=0;record_index=0
 with gzip.open(shard,'rb') as f,temp.open('w') as out:
  while True:
   header=f.read(8)
   if not header:break
   n=struct.unpack('<Q',header)[0];assert len(f.read(4))==4
   b=f.read(n);assert len(b)==n and len(f.read(4))==4
   ex=Example.FromString(b).features.feature
   ep=int(ex['episode_id'].int64_list.value[0]);record_index+=1
   if ep not in wanted:continue
   actions=[json.loads(x) for x in ex['actions'].bytes_list.value]
   shots=ex['screenshots'].bytes_list.value
   lows=[x.decode() for x in ex['step_instructions'].bytes_list.value]
   assert len(shots)==len(actions)+1 and len(lows)==len(actions)
   goal=ex['goal'].bytes_list.value[0].decode()
   ws=list(ex['screenshot_widths'].int64_list.value);hs=list(ex['screenshot_heights'].int64_list.value)
   episode=dict(episode_id=str(ep),split=splitmap.get(ep,'unknown'),goal=goal,actions=actions,step_instructions=lows,
    screenshot_widths=ws,screenshot_heights=hs,screenshot_sha256=[hashlib.sha256(x).hexdigest() for x in shots],
    original_shard=str(shard),original_record_index=record_index-1,original_record_sha256=hashlib.sha256(b).hexdigest())
   epdest=OUT/(str(ep)+'.json');atom(epdest,episode)
   for step,uid in wanted[ep].items():
    row=dict(id=uid,source='android_control',episode_id=str(ep),step=step,split=episode['split'],annotation_path=str(epdest),
      annotation_sha256=hashlib.sha256(epdest.read_bytes()).hexdigest(),original_shard=str(shard),original_record_index=record_index-1,
      original_record_sha256=episode['original_record_sha256'],approved_for_training=False)
    if step>=len(actions):row['status']='not_an_original_action_step'
    else:
     row.update(status='raw_record_extracted',goal=goal,low_instruction=lows[step],original_action=actions[step],
      current_raw_sha256=episode['screenshot_sha256'][step],next_raw_sha256=episode['screenshot_sha256'][step+1],
      width=ws[step],height=hs[step],previous_actions=actions[max(0,step-4):step])
    out.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
   episodes+=1
 temp.replace(dest)
 result=dict(shard=str(shard),size=st.st_size,mtime_ns=st.st_mtime_ns,rows=count,episodes=episodes,records_scanned=record_index,output=str(dest))
 atom(done,result);return result
with (P/'ac-tfrecord-extract.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 shards=sorted(RAW.glob('android_control-*-of-*'));assert len(shards)==20
 completed=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for result in pool.map(worker,shards):
   completed.append(result)
   report=dict(completed_shards=len(completed),total_shards=len(shards),rows=sum(x['rows'] for x in completed),episodes=sum(x['episodes'] for x in completed))
   atom(OUT/'progress.json',report);print(json.dumps(report),flush=True)
 with (P/'artifacts/ac-official-tfrecord-records.jsonl').open('w') as out:
  for shard in shards:
   with (OUT/(shard.name+'.jsonl')).open() as f:
    for line in f:out.write(line)
 report.update(complete=True,requested_rows=sum(map(len,wanted.values())),official_split_sha256=hashlib.sha256((RAW/'splits.json').read_bytes()).hexdigest(),training_ready=False)
 atom(OUT/'complete.json',report);print(json.dumps(report),flush=True)
