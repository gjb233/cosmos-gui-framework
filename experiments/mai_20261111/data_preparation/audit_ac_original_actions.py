"""CPU audit of AndroidControl published targets against existing original-derived records.
No training approval is inferred from filenames or a successful syntax parse.
"""
import collections,concurrent.futures,hashlib,json,pathlib,re,fcntl
from PIL import Image
P=pathlib.Path(__file__).resolve().parent
R=pathlib.Path('/rscratch/wzzheng/experiments/gui-libra-low-terminal-cefm500-a22-20261014-v1')
S=pathlib.Path('/rscratch/wzzheng/data/androidcontrol/raw/splits.json')
def rows(p):return map(json.loads,p.open())
native={(r['episode_id'],r['step']):r for r in rows(R/'train.jsonl') if not r['episode_id'].startswith('aitz:')}
actions={r['id']:r for r in rows(P/'artifacts/action-pair-audit.jsonl') if r['source']=='android_control'}
images={r['id']:r for r in rows(P/'artifacts/image-decode-audit.jsonl') if r['source']=='android_control'}
split=json.loads(S.read_text());train=set(map(str,split['train']));other=set(map(str,split['test']+split['validation']))
held=json.loads((P/'heldout.json').read_text());heldeps={tuple(v) for v in held['episodes']};heldsha=set(held['image_sha256'])
def semantic(a,n):
 k=n['type'];expected={'click':'Click','long_press':'LongPress','input_text':'Write','scroll':'Scroll','open_app':'OpenAPP','navigate_home':'NavigateHome','navigate_back':'NavigateBack','navigate_enter':'KeyboardPress','wait':'Wait','terminate':'Terminate'}.get(k)
 if a['action_type']!=expected:return 'type_changed'
 if k in ('click','long_press'):
  point=a.get('point_2d');target=[round(1000*n['x']),round(1000*n['y'])]
  return 'exact_rounded_coordinate' if point==target else 'coordinate_changed'
 if k in ('input_text','scroll','open_app'):
  value=n[{'input_text':'text','scroll':'direction','open_app':'app_name'}[k]]
  return 'exact_parameter' if a.get('value')==value else 'parameter_changed'
 if k in ('navigate_home','navigate_back'):return 'exact_navigation'
 if k=='wait':return 'wait_duration_not_in_original'
 if k=='navigate_enter':return 'exact_enter' if str(a.get('value')).lower() in ('enter','return') else 'parameter_changed'
 if k=='terminate':return 'synthetic_terminal_requires_separate_review'
 return 'unsupported_original_action'
def check(r):
 ep,st=re.fullmatch(r'(\d+)/screenshot_(\d+)\.png',r['image']).groups();st=int(st)
 out=dict(id=r['id'],source=r['source'],episode_id=ep,step=st,official_train=ep in train and ep not in other,heldout_episode=('android_control',ep) in heldeps,heldout_image=images[r['id']]['sha256'] in heldsha,approved_for_training=False,fm_approved=False)
 n=native.get((ep,st))
 if n is None:return dict(out,status='original_derived_record_missing')
 im=images[r['id']]['image']
 with Image.open(im) as x,Image.open(n['current_image']) as y:
  same_size=x.size==y.size
  out['current_pixel_identity_verified']=same_size and x.convert('RGB').tobytes()==y.convert('RGB').tobytes()
  out['published_size']=list(x.size);out['original_size']=list(y.size)
 out['action_status']=semantic(actions[r['id']]['action'],n['action'])
 out['goal']=n['goal'];out['low_instruction']=n['instruction'];out['original_action']=n['action'];out['original_image']=n['current_image']
 # Existing original-derived records supply the next image; verify its byte content and path now.
 nxt=n.get('next_image')
 out['original_next_exists']=bool(nxt and pathlib.Path(nxt).is_file())
 if out['original_next_exists']:
  out['next_image']=nxt;out['next_image_sha256']=hashlib.sha256(pathlib.Path(nxt).read_bytes()).hexdigest()
  out['next_heldout_image']=out['next_image_sha256'] in heldsha
 out['status']='checked'
 return out
with (P/'ac-original-action-audit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 counts=collections.Counter();results=[]
 records=[r for r in rows(P/'artifacts/inventory.jsonl') if r['source']=='android_control']
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool,(P/'artifacts/ac-original-action-audit.jsonl').open('w') as f:
  for i,r in enumerate(pool.map(check,records)):
   f.write(json.dumps(r,ensure_ascii=False)+'\n');counts[r['status']]+=1
   for key in ['current_pixel_identity_verified','official_train','heldout_episode','heldout_image','original_next_exists','next_heldout_image']:
    if r.get(key):counts[key]+=1
   if 'action_status' in r:counts['action:'+r['action_status']]+=1
   if i%500==0:
    f.flush();(P/'artifacts/ac-original-action-progress.json').write_text(json.dumps(dict(done=i+1,total=len(records),counts=counts)))
 report=dict(counts=counts,rows=len(records),original_derived_manifest=str(R/'train.jsonl'),manifest_sha256=hashlib.sha256((R/'train.jsonl').read_bytes()).hexdigest(),official_split_path=str(S),official_split_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),training_ready=False,note='Pixel identity and original-derived action/split audit. Original TFRecord lineage and instruction/description semantic review remain required; no blanket approval.')
 (P/'artifacts/ac-original-action-report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report),flush=True)
