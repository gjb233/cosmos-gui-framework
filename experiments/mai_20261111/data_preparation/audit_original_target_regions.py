import pathlib,json,collections
P=pathlib.Path(__file__).resolve().parent
images={r['id']:r for r in map(json.loads,(P/'artifacts/image-decode-audit.jsonl').open())}
counts=collections.defaultdict(collections.Counter)
out=P/'artifacts/original-target-region-diagnostic.jsonl'
with out.open('w') as f:
 for name,source in [('mind2web-original-action-candidates.jsonl','mind2web'),('amex-original-action-candidates.jsonl','amex')]:
  for r in map(json.loads,(P/'artifacts'/name).open()):
   a=r.get('published_action',{});pt=a.get('point_2d')
   if a.get('action_type')!='Click' or not isinstance(pt,list) or len(r.get('matches',[]))!=1:continue
   raw=r['matches'][0];boxes=[];err=None
   if source=='amex':
    n=raw['original_step'];w,h=n['device_dim'];reg=n.get('interest_region')
    if reg and len(reg)==2:
     try:
      x1,y1=reg[0];x2,y2=reg[1]
      if x2>x1 and y2>y1:boxes.append([x1,y1,x2,y2])
     except Exception:err='invalid_region'
    size_source='official_device_dim'
   else:
    im=images[r['id']]
    from PIL import Image
    with Image.open(im['image']) as img:w,h=img.size
    size_source='published_screenshot_size_not_yet_original_identity_verified'
    for candidate in raw['positive_candidates']:
     try:
      attr=json.loads(candidate['attributes']);x,y,bw,bh=map(float,attr['bounding_box_rect'].split(','))
      if bw>0 and bh>0:boxes.append([x,y,x+bw,y+bh])
     except (ValueError,KeyError,TypeError):err='bbox_unavailable'
   px,py=pt[0]*w/1000,pt[1]*h/1000
   contained=[b for b in boxes if b[0]<=px<=b[2] and b[1]<=py<=b[3]]
   status='inside_original_positive_region' if contained else 'outside_original_positive_regions' if boxes else 'no_original_region'
   counts[source][status]+=1
   row=dict(id=r['id'],source=source,status=status,point=pt,size=[w,h],size_source=size_source,boxes=boxes,contained=contained,approved_for_training=False,fm_approved=False)
   f.write(json.dumps(row)+'\n')
report=dict(counts=counts,training_ready=False,note='Region containment is diagnostic evidence, not semantic/trajectory approval. Mind2Web raw HTML bbox viewport/crop alignment still requires verification.')
(P/'artifacts/original-target-region-diagnostic-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
