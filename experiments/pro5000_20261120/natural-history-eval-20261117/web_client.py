from pathlib import Path
import sys,os,json,time,ast,math,argparse,urllib.request,concurrent.futures,threading,itertools
arg=argparse.ArgumentParser();arg.add_argument('--step',type=int,required=True);arg.add_argument('--ports',required=True);arg.add_argument('--concurrency',type=int,default=64);arg.add_argument('--limit',type=int,default=0)
args=arg.parse_args();ports=[int(p) for p in args.ports.split(',')]
R=Path('/data/mai/evaluation/natural-history-20261117')/os.environ['PROMPT_ARM'];A=Path('/data/mai/evaluation/assets-20261113')
out=(R/'smoke' if args.limit else R)/f'step-{args.step:06d}';out.mkdir(parents=True,exist_ok=True)
counter=itertools.count();lock=threading.Lock()
def generate(messages):
 with lock:port=ports[next(counter)%len(ports)]
 body=json.dumps(dict(model='eval',messages=messages,max_tokens=2048,temperature=0,seed=42)).encode()
 req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=1800) as response:result=json.load(response)
 choice=result['choices'][0]
 return choice['message'].get('content') or '',result['usage']['completion_tokens'],choice['finish_reason']=='length',result['usage'].get('prompt_tokens')
for bench in ('grounding','mind2web'):
 d=A/bench;sys.path.insert(0,str(d))
 for module in ('native','native_protocol','evaluation_protocol','protocol','official_metrics'):sys.modules.pop(module,None)
 import prompt_adapter as native
 if bench=='grounding':from protocol import plan_ans
 else:
  from official_metrics import metrics
  tree=ast.parse((d/'client.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score']
  ns=dict(parse=native.convert,metrics=metrics,math=math);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-score>','exec'),ns);score=ns['score']
 rows=[json.loads(l) for l in (d/'manifest.jsonl').open()]
 if bench=='grounding':rows=[r for r in rows if r['dataset'] in ('androidcontrol-high','androidcontrol-low')]
 if args.limit:rows=rows[:args.limit]
 dest=out/f'{bench}-predictions-0.jsonl';done={json.loads(l)['id'] for l in dest.open()} if dest.exists() else set()
 def predict(row):
  start=time.monotonic()
  if bench=='grounding':
   text,n,truncated,prompt_tokens=generate(native.ground_messages(row));a=native.convert(text);plan,ans,error=plan_ans(row,json.dumps(a or {}))
   result=dict(id=row['id'],dataset=row['dataset'],response=text,output_tokens=n,prompt_tokens=prompt_tokens,truncated=truncated,plan=plan,ans=ans,point_parse_error=error,native_parse_error=a is None)
  else:
   traces=[]
   for block in range(len(row['images'])):
    messages=native.m2_messages(row,block);text,n,truncated,prompt_tokens=generate(messages);traces.append(dict(block=block,text=text,output_tokens=n,prompt_tokens=prompt_tokens,truncated=truncated,prompt=[dict(role=m['role'],content=[x for x in m['content'] if x['type']=='text']) for m in messages]))
    a=native.convert(text)
    if not(a and a['action_type'].lower()=='scroll' and a['value'].lower()=='down'):break
   result=dict(id=row['id'],split=row['split'],traces=traces,**score(row,text,block))
  result['seconds']=time.monotonic()-start;return result
 pending=[r for r in rows if r['id'] not in done];failures=[];start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
  futures={pool.submit(predict,row):row['id'] for row in pending}
  for f in concurrent.futures.as_completed(futures):
   try:result=f.result()
   except Exception as e:
    failures.append(dict(id=futures[f],error=repr(e)))
    with (out/'request-errors.jsonl').open('a') as log:log.write(json.dumps(failures[-1])+'\n')
    continue
   with dest.open('a') as log:log.write(json.dumps(result,ensure_ascii=False)+'\n')
   done.add(result['id']);(out/'progress.json').write_text(json.dumps(dict(benchmark=bench,completed=len(done),total=len(rows),elapsed_seconds=time.monotonic()-start,concurrency=args.concurrency,ports=ports,time=time.time())))
   if len(done)%50==0:print('EVALUATED',args.step,bench,len(done),len(rows),flush=True)
 assert not failures,failures[:3]
 assert done=={r['id'] for r in rows}
(out/'complete.json').write_text(json.dumps(dict(completed=True,step=args.step,concurrency=args.concurrency,ports=ports,engine='vllm',smoke=bool(args.limit))))
