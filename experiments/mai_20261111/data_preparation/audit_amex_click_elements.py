import pathlib,json,zipfile,collections,hashlib
P=pathlib.Path(__file__).resolve().parent
zpath=P/'originals/amex-element-anno.zip';counts=collections.Counter()
with zipfile.ZipFile(zpath) as z,(P/'artifacts/amex-click-element-alignment.jsonl').open('w') as f:
 for r in map(json.loads,(P/'artifacts/amex-original-action-candidates.jsonl').open()):
  if len(r.get('matches',[]))!=1:continue
  n=r['matches'][0]['original_step'];a=r.get('published_action',{})
  if n['action']!='TAP' or a.get('action_type')!='Click' or not isinstance(a.get('point_2d'),list):continue
  entry='element_anno/'+pathlib.Path(n['image_path']).stem+'.json'
  result=dict(id=r['id'],episode_id=r['episode_id'],element_entry=entry,approved_for_training=False,fm_approved=False)
  try:
   b=z.read(entry);ann=json.loads(b);ox,oy=n['touch_coord'];w,h=n['device_dim'];px,py=[a['point_2d'][i]*n['device_dim'][i]/1000 for i in range(2)]
   candidates=[]
   for el in ann.get('clickable_elements',[]):
    x1,y1,x2,y2=el['bbox']
    if x2>x1 and y2>y1 and x1<=ox<=x2 and y1<=oy<=y2:
     candidates.append(((x2-x1)*(y2-y1),el))
   candidates.sort(key=lambda x:x[0])
   smallest=[el for area,el in candidates if area==candidates[0][0]] if candidates else []
   shared=[el for el in smallest if el['bbox'][0]<=px<=el['bbox'][2] and el['bbox'][1]<=py<=el['bbox'][3]]
   status='same_unique_smallest_original_clickable_element' if len(smallest)==1 and len(shared)==1 else 'ambiguous_smallest_element' if shared else 'different_element_or_boundary' if smallest else 'no_clickable_element_at_original_touch'
   result.update(status=status,raw_touch=[ox,oy],published_touch_pixels=[px,py],matched_elements=shared,original_smallest_elements=smallest,element_annotation_sha256=hashlib.sha256(b).hexdigest())
  except Exception as e:result.update(status='error',error=repr(e))
  counts[result['status']]+=1;f.write(json.dumps(result,ensure_ascii=False)+'\n')
report=dict(counts=counts,rows=sum(counts.values()),training_ready=False,note='Same smallest original clickable bbox is evidence of target equivalence, not full action/description/next approval. All records remain unapproved here.')
(P/'artifacts/amex-click-element-alignment-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
