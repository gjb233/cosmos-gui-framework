"""Conservative original-action diagnostic; never grants FM approval."""
import json, pathlib, collections, re, math
P=pathlib.Path('/rscratch/wzzheng/experiments/mixed-sft-preparation-20261028-v1')
def rows(n):return map(json.loads,open(P/'artifacts'/n))
targets={r['id']:r for r in rows('action-pair-audit.jsonl')}
def compare(r,a):
 n=r['original_records'][0];k=a['action_type'];v=a.get('value');point=a.get('point_2d')
 if r['source']=='coat-terminal':
  typ=n['result_action_type']
  if typ==3:return 'exact_text_match' if k=='Write' and v==n['result_action_text'] else 'type_or_text_changed'
  mapping={5:'NavigateBack',6:'NavigateHome',7:'KeyboardPress',10:'Terminate'}
  if typ in mapping:
   return 'type_match_parameters_pending' if k==mapping[typ] else 'action_type_changed'
  if typ!=4:return 'original_type_needs_mapping'
  start=json.loads(n['result_touch_yx']);end=json.loads(n['result_lift_yx'])
  if start!=end:return 'gesture_endpoints_not_represented_in_target'
  if k!='Click':return 'action_type_changed'
  if not isinstance(point,list):return 'missing_target_coordinates'
  delta=max(abs(point[0]-start[1]*1000),abs(point[1]-start[0]*1000))
  return 'rounded_point_match' if delta<=1 else 'point_changed'
 acts=n['actions_label']
 if len(acts)!=1:return 'compound_original_step'
 raw=acts[0];name=raw['name']
 if name in ('input','answer'):
  wanted='Write' if name=='input' else 'Answer'
  return 'exact_text_match' if k==wanted and v==raw['text'] else 'type_or_text_changed'
 if name=='click':
  if k!='Click':return 'action_type_changed'
  xy=[float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',raw['point']['related'])]
  if not isinstance(point,list) or len(xy)!=2:return 'missing_target_coordinates'
  return 'rounded_point_match' if max(abs(point[i]-xy[i]*1000) for i in (0,1))<=1 else 'point_changed'
 if name=='enter':return 'key_match' if k=='KeyboardPress' and v=='Enter' else 'type_or_key_changed'
 if name=='scroll':
  return 'scroll_distance_not_represented_in_target' if k=='Scroll' else 'action_type_changed'
 return 'original_action_not_equivalent_to_supported_target'
counts=collections.defaultdict(collections.Counter);examples=collections.defaultdict(list)
with (P/'artifacts/original-action-transition-review.jsonl').open('w') as f:
 for r in rows('original-trajectory-candidates.jsonl'):
  t=targets[r['id']]
  status=r['status']
  if status=='unique_original_content':
   status=compare(r,t['action']) if t.get('parsed') else 'target_parse_error'
  result=dict(id=r['id'],source=r['source'],episode=r.get('episode_id'),step=r.get('step'),status=status,next_image=r.get('next_image_candidate'),fm_approved=False,
   reason='Original-action diagnostics do not certify split, screenshot identity, executor parameters or visual causality.')
  counts[r['source']][status]+=1
  if r.get('next_image_candidate'):counts[r['source']]['has_next_image']+=1
  key=r['source']+':'+status
  if len(examples[key])<2:examples[key].append(dict(result,target=t.get('action'),original=r['original_records'][:1],current_image=r['current_image']))
  f.write(json.dumps(result)+'\n')
report=dict(counts=dict(counts),training_ready=False,fm_approved=0,scope='All indexed CoAT and GUIAct rows. Point matches tolerate 1/1000 quantization, not visual certification.')
(P/'artifacts/original-action-transition-review-report.json').write_text(json.dumps(report,indent=2))
(P/'artifacts/original-action-transition-review-examples.json').write_text(json.dumps(examples))
print(json.dumps(report,indent=2))
