from pathlib import Path
import sys,json,base64,math,concurrent.futures,hashlib
E=Path(__file__).resolve().parent;V=E
cfg=json.loads((E/'config.json').read_text())
sys.path.insert(0,cfg['q']);sys.path.insert(0,cfg['source']);sys.path.insert(0,str(E))
from native import ground_messages,convert,ground_point
from protocol import plan_ans
from official_eval import process_file
from openai import OpenAI
api=OpenAI(base_url='http://127.0.0.1:25452/v1',api_key='EMPTY',timeout=3600,max_retries=0)
rows=[json.loads(l) for l in (E/'manifest.jsonl').open()]
done={r['id']:r for r in map(json.loads,(V/'preserved.jsonl').open())}
dest=V/'predictions.jsonl'
if dest.exists():
 for l in dest.open():
  r=json.loads(l);done[r['id']]=r
def predict(row):
 m=ground_messages(row)
 response=api.chat.completions.create(model='eval',messages=m,max_tokens=512,temperature=0,seed=42,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
 text=response.choices[0].message.content or ''
 result=dict(id=row['id'],dataset=row['dataset'],response=text,output_tokens=response.usage.completion_tokens,truncated=response.choices[0].finish_reason=='length',engine='mai-native-nothinking-fp16-concurrency16')
 if row['dataset'].startswith('androidcontrol'):
  a=convert(text);plan,ans,error=plan_ans(row,json.dumps(a or {}));result.update(plan=plan,ans=ans,point_parse_error=error,native_parse_error=a is None)
 else:
  p=ground_point(text)
  parsed=dict(action_type='click',point_2d=p)
  valid=p is not None and all(0<=x<=1000 for x in p);hit=False
  if valid and parsed['action_type'].lower()=='click':
   x,y=p[0]*row['width']/1000,p[1]*row['height']/1000;a,b,c,d=row['bbox'];hit=a<=x<=c and b<=y<=d
  result.update(parsed=parsed,correct=hit)
 return result
pending=[r for r in rows if r['id'] not in done]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 futures={pool.submit(predict,r):r['id'] for r in pending}
 for f in concurrent.futures.as_completed(futures):
  try:result=f.result()
  except Exception as exc:
   with (V/'request-errors.jsonl').open('a') as err:err.write(json.dumps(dict(id=futures[f],error=repr(exc)))+'\n')
   print('REQUEST_FAILED',futures[f],repr(exc),flush=True)
   continue
  with dest.open('a') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n')
  done[result['id']]=result
  (V/'progress.json').write_text(json.dumps(dict(completed_total=len(done),total=len(rows),last_id=result['id'])))
  print('EVALUATED',len(done),len(rows),result['id'],flush=True)
assert set(done)=={r['id'] for r in rows}
(E/'predictions-merged.jsonl').write_text(''.join(json.dumps(done[r['id']],ensure_ascii=False)+'\n' for r in rows))
summary={}
for dataset in cfg['datasets']:
 rr=[done[r['id']] for r in rows if r['dataset']==dataset]
 if dataset.startswith('androidcontrol'):
  d=E/dataset;d.mkdir(exist_ok=True)
  for field in ['plan','ans']:(d/(field+'.jsonl')).write_text(''.join(json.dumps(r[field])+'\n' for r in rr))
  score=dict(process_file(str(E/'samples.json'),str(d/'plan.jsonl'),str(d/'ans.jsonl')))
  score['accuracy']=score['correct_steps']/score['total_steps']
 else:score=dict(total_steps=len(rr),correct_steps=sum(r['correct'] for r in rr),accuracy=sum(r['correct'] for r in rr)/len(rr))
 score['truncated']=sum(r['truncated'] for r in rr);summary[dataset]=score
(E/'results.json').write_text(json.dumps(summary,indent=2))
(E/'complete.json').write_text(json.dumps(dict(completed=True,checkpoint=json.loads((E/'config.json').read_text())['checkpoint'],samples=len(done),results=summary),indent=2))
