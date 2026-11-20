import json,pathlib,collections,ast
from audit_upstream_actions import commands
P=pathlib.Path(__file__).resolve().parent
images={r['id']:r for r in map(json.loads,(P/'artifacts/image-decode-audit.jsonl').open())}
targets={r['id']:r for r in map(json.loads,(P/'artifacts/action-pair-audit.jsonl').open())}
counts=collections.defaultdict(collections.Counter)
with (P/'artifacts/upstream-coordinate-formula-diagnostic.jsonl').open('w') as out:
 for r in map(json.loads,(P/'artifacts/upstream-action-alignment.jsonl').open()):
  a=targets[r['id']].get('action',{});pt=a.get('point_2d')
  if a.get('action_type') not in ('Click','LongPress') or not isinstance(pt,list) or len(r['matches'])!=1:continue
  try:
   calls=[c for text in r['matches'][0]['commands'] for c in commands(text)]
   if len(calls)!=1 or calls[0][0] not in ('pyautogui.click','mobile.long_press'):continue
   kw=calls[0][2];xy=[kw['x'],kw['y']]
   if not all(0<=v<=1 for v in xy):continue
  except Exception:continue
  im=images[r['id']];dims=[im['width'],im['height']];fits=[]
  rules={}
  for scale in [999,1000]:
   rules[str(scale)+'_floor']=[int(v*scale) for v in xy]
   rules[str(scale)+'_round']=[round(v*scale) for v in xy]
  for minus in [0,1]:
   for rounding in ['floor','round']:
    pix=[int(xy[i]*(dims[i]-minus)) if rounding=='floor' else round(xy[i]*(dims[i]-minus)) for i in range(2)]
    for out_round in ['floor','round']:
     key='pixel'+str(minus)+'_'+rounding+'_'+out_round
     rules[key]=[int(pix[i]/dims[i]*1000) if out_round=='floor' else round(pix[i]/dims[i]*1000) for i in range(2)]
  fits=[k for k,v in rules.items() if v==pt]
  counts[r['source']]['rows']+=1
  for k in fits:counts[r['source']][k]+=1
  if not fits:counts[r['source']]['no_formula_match']+=1
  out.write(json.dumps(dict(id=r['id'],source=r['source'],upstream_point=xy,published_point=pt,size=dims,matching_formulas=fits,approved=False))+'\n')
report=dict(counts=counts,approval=False,note='Hypothesis diagnostics across original intermediate commands and screenshot sizes. A numerical match alone is not verified conversion provenance.')
(P/'artifacts/upstream-coordinate-formula-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
