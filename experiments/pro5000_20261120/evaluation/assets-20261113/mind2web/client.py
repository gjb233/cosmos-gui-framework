from pathlib import Path
import sys,json,hashlib,base64,concurrent.futures,time,math
R=Path(__file__).resolve().parent
cfg=json.loads((R/'config.json').read_text())
sys.path.insert(0,cfg['q'])
from native import m2_messages,convert
from official_metrics import metrics
from openai import OpenAI
def messages(row,block):return m2_messages(row,block)
def parse(text):return convert(text)
def score(row,text,block):
 a=parse(text);p=a.get('point_2d') if a else None
 valid=isinstance(p,list) and len(p)==2 and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1000 for x in p)
 entry=dict(row,response='<answer>'+text+'</answer>',bbox=row['target_blocks'].get(str(block),[]),scale=1.0,image_width=row['images'][block]['width'],image_height=row['images'][block]['height'],ans_block=block,gpt_action=a['action_type'] if a else 'no action',gpt_value=a['value'] if a else 'no input text',point_2d=p if valid else [-100000,-100000])
 v=metrics([entry])
 return dict(element_acc=int(v['element_acc']),operation_f1=float(v['operation_f1']),correct=bool(v['step_acc']),error=None if a else 'invalid_json')
def predict(row):
 traces=[]
 for block in range(len(row['images'])):
  m=messages(row,block)
  resp=api.chat.completions.create(model='eval',messages=m,max_tokens=512,temperature=0,seed=42,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
  text=resp.choices[0].message.content or ''
  traces.append(dict(block=block,text=text,prompt_sha256=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest(),truncated=resp.choices[0].finish_reason=='length'))
  a=parse(text)
  if not(a and a['action_type'].lower()=='scroll' and a['value'].lower()=='down'):break
 return dict(id=row['id'],split=row['split'],traces=traces,**score(row,text,block))
def aggregate(rows,done):
 def group(rr):
  n=len(rr);pp=[done[x['id']] for x in rr]
  return dict(total=n,correct=sum(x['correct'] for x in pp),step_success_rate=sum(x['correct'] for x in pp)/n,element_accuracy=sum(x['element_acc'] for x in pp)/n,operation_f1=sum(x['operation_f1'] for x in pp)/n,parse_errors=sum(x['error'] is not None for x in pp),truncated_requests=sum(t['truncated'] for x in pp for t in x['traces']))
 return {'overall':group(rows),**{'cross_'+s:group([r for r in rows if r['split']==s]) for s in ['task','website','domain']}}
if __name__=='__main__':
 rows=[json.loads(l) for l in (R/'manifest.jsonl').open()]
 done={};dest=R/'predictions.jsonl'
 if dest.exists():done={x['id']:x for x in map(json.loads,dest.open())}
 api=OpenAI(base_url='http://127.0.0.1:25452/v1',api_key='EMPTY',timeout=3600,max_retries=0)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(predict,r):r['id'] for r in rows if r['id'] not in done}
  for f in concurrent.futures.as_completed(futures):
   try:result=f.result()
   except Exception as exc:
    with (R/'request-errors.jsonl').open('a') as err:err.write(json.dumps(dict(id=futures[f],error=repr(exc)))+'\n')
    print('REQUEST_FAILED',futures[f],repr(exc),flush=True)
    continue
   with dest.open('a') as out:out.write(json.dumps(result,ensure_ascii=False)+'\n')
   done[result['id']]=result
   (R/'progress.json').write_text(json.dumps(dict(completed=len(done),total=len(rows),last_id=result['id'],time=time.time())))
   print('EVALUATED',len(done),len(rows),result['id'],flush=True)
 assert set(done)=={r['id'] for r in rows}
 (R/'predictions-ordered.jsonl').write_text(''.join(json.dumps(done[r['id']],ensure_ascii=False)+'\n' for r in rows))
 result=aggregate(rows,done)
 (R/'results.json').write_text(json.dumps(result,indent=2))
 (R/'complete.json').write_text(json.dumps(dict(completed=True,checkpoint=200,samples=len(rows),results=result),indent=2))
