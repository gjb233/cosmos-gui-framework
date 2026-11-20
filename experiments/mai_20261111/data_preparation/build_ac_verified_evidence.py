"""Build conservative AndroidControl evidence from raw TFRecords and pixel audit.
Only exact original command parameters are admitted. Other records remain reviewable.
"""
import json,hashlib,pathlib,collections,shutil
from mixed_prompt import target
P=pathlib.Path(__file__).resolve().parent
def rows(p):return map(json.loads,p.open())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert (P/'originals/androidcontrol-selected/complete.json').exists()
assert (P/'artifacts/ac-original-action-report.json').exists()
aud={r['id']:r for r in rows(P/'artifacts/ac-original-action-audit.jsonl')}
actions={r['id']:r['action'] for r in rows(P/'artifacts/action-pair-audit.jsonl') if r.get('parsed') and r['source']=='android_control'}
images={r['id']:r for r in rows(P/'artifacts/image-decode-audit.jsonl') if r['source']=='android_control'}
held=json.loads((P/'heldout.json').read_text());heldsha=set(held['image_sha256']);heldeps={tuple(x) for x in held['episodes']}
outdir=P/'evidence';outdir.mkdir(exist_ok=True)
D=pathlib.Path(json.loads((P/'mixture.json').read_text())['image_root'])
accepted=[];rejected=[];counts=collections.Counter()
names={'click':'Click','long_press':'LongPress','input_text':'Write','scroll':'Scroll','open_app':'OpenAPP','navigate_home':'NavigateHome','navigate_back':'NavigateBack','navigate_enter':'KeyboardPress','wait':'Wait'}
def canonical(a,width,height):
 k=a['action_type'];value={'input_text':a.get('text'),'scroll':a.get('direction'),'open_app':a.get('app_name'),'navigate_enter':'Enter'}.get(k,'None')
 point=[round(a['x']/width*1000),round(a['y']/height*1000)] if k in ('click','long_press') else 'None'
 return dict(action_type=names[k],action_target='None',value=value,point_2d=point)
for raw in rows(P/'artifacts/ac-official-tfrecord-records.jsonl'):
 uid=raw['id'];a=aud[uid];t=actions[uid];reasons=[]
 if raw['status']!='raw_record_extracted':reasons.append('not_original_action_step')
 if raw['split']!='train' or not a.get('official_train'):reasons.append('not_official_train')
 if a.get('heldout_episode') or a.get('heldout_image'):reasons.append('heldout_overlap')
 if not a.get('current_pixel_identity_verified'):reasons.append('current_identity_not_verified')
 if a.get('action_status') not in ('exact_navigation','exact_parameter','exact_rounded_coordinate','exact_enter'):reasons.append(a.get('action_status','missing_original_derived_record'))
 if reasons:
  rejected.append(dict(id=uid,reasons=reasons));counts.update(reasons);continue
 # Bind the earlier decoded-pixel audit to the actual screenshot bytes inside the official record.
 assert sha(images[uid]['image'])==images[uid]['sha256'], 'published image changed since audit'
 if sha(a['original_image'])!=raw['current_raw_sha256']:raise ValueError('original image lineage mismatch '+uid)
 if a['goal']!=raw['goal'] or a['low_instruction']!=raw['low_instruction']:raise ValueError('instruction lineage mismatch '+uid)
 if a['published_size']!=[raw['width'],raw['height']]:
  rejected.append(dict(id=uid,reasons=['raw_image_dimension_conflict']));counts['raw_image_dimension_conflict']+=1;continue
 expected=canonical(raw['original_action'],raw['width'],raw['height'])
 if any(t[k]!=expected[k] for k in ('action_type','value','point_2d')):
  rejected.append(dict(id=uid,reasons=['raw_parameter_or_optional_field_changed']));counts['raw_parameter_or_optional_field_changed']+=1;continue
 ep=json.loads(pathlib.Path(raw['annotation_path']).read_text());st=raw['step']
 history=[dict(episode_id=raw['episode_id'],step=j,action=canonical(ep['actions'][j],ep['screenshot_widths'][j],ep['screenshot_heights'][j])) for j in range(max(0,st-4),st)]
 nxt=None;nextproof=None
 if a.get('original_next_exists') and not a.get('next_heldout_image'):
  source=pathlib.Path(a['next_image'])
  if sha(source)!=raw['next_raw_sha256']:raise ValueError('next image lineage mismatch '+uid)
  dest=D/'audited-originals/android-control'/raw['episode_id']/(str(st+1)+'.png')
  dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copyfile(source,dest)
  assert sha(dest)==raw['next_raw_sha256']
  nxt=str(dest);nextproof=raw['next_raw_sha256']
 views={level:dict(action=t,direct=target(t),instruction_consistency_verified=True,verification='official step action parameters exact; current image RGB identical; original task/step instruction') for level,text in [('low',raw['low_instruction']),('high',raw['goal'])] if text.strip()}
 record=dict(id=uid,source='android_control',episode_id=raw['episode_id'],step=st,split='train',split_verified=True,current_image_identity_verified=True,
  annotation_path=raw['annotation_path'],annotation_sha256=raw['annotation_sha256'],original_shard=raw['original_shard'],original_record_sha256=raw['original_record_sha256'],
  official_split_sha256=sha('/rscratch/wzzheng/data/androidcontrol/raw/splits.json'),current_image=images[uid]['image'],current_original_sha256=raw['current_raw_sha256'],
  instruction_level='low',instruction=raw['low_instruction'],low_instruction=raw['low_instruction'],goal=raw['goal'],history=history,targets=views,
  terminal=False,next_image=nxt,next_step=st+1 if nxt else None,next_episode_id=raw['episode_id'] if nxt else None,next_original_sha256=nextproof,
  action_alignment_verified=True,approved_for_training=True,approval_scope='exact original AndroidControl command; no rewritten action or derived Low approval')
 accepted.append(record)
for filename,data in [('android-control.jsonl',accepted),('android-control-review-required.jsonl',rejected)]:
 with (outdir/filename).open('w') as f:
  for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
report=dict(source='android_control',total=len(aud),approved=len(accepted),fm_transitions=sum(bool(r['next_image']) for r in accepted),review_required=len(rejected),reasons=counts,
 inputs={str(p):sha(p) for p in [P/'artifacts/ac-official-tfrecord-records.jsonl',P/'artifacts/ac-original-action-audit.jsonl',P/'heldout.json']},all_sources_ready=False)
(outdir/'android-control-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
