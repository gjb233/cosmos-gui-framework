from pathlib import Path
import ast, json, re, math, base64, hashlib, collections, asyncio, random, os, io, types, copy
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent
PLAN=(R/'upstream/guilibra_plan.py').read_text()
EVAL=(R/'upstream/eval_rl.py').read_text()
def extract(source, names):
 tree=ast.parse(source)
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
 return compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-upstream>','exec')
P=dict(os=os,re=re,json=json,base64=base64,Image=Image,io=io,random=random)
exec(extract(PLAN,{'system_prompt','question_description','encode_image','process_row'}),P)
import string,unicodedata
E=dict(re=re,np=np,collections=collections,os=os,Image=Image,string=string,unicodedata=unicodedata)
exec(extract(EVAL,{'_PUNC_RE','_WS_RE','normalize_text','calculate_f1','is_output_inside_bbox','extract_action','extract_input_text','extract_coord','get_metrics_with_prediction'}),E)
E['args']=types.SimpleNamespace(relative_coord=1,block_image_dir='')
def configure(job):
 # Both registered model configs are qwen3_vl. Use an architecture-bearing model identifier so upstream selects its Qwen3 template.
 P['args']=types.SimpleNamespace(model='Qwen3-VL-'+job['name'],add_template=1,reasoning=1,temperature=0.0)
def metrics(entries):
 return E['get_metrics_with_prediction'](entries)
def predict(client,model,row):
 Q=dict(P)
 Q['process_row']=types.FunctionType(P['process_row'].__code__,Q,P['process_row'].__name__,P['process_row'].__defaults__,P['process_row'].__closure__)
 for im in row['images']:
  assert hashlib.sha256(Path(im['path']).read_bytes()).hexdigest()==im['sha256']
 Q['block_image_dir']=str(Path(row['images'][0]['path']).parent.parent)
 Q['client']=client;Q['model_name']=model
 traces=[]
 async def response(client,model,messages,temperature=0.0):
  resp=client.chat.completions.create(model=model,messages=messages,temperature=temperature,max_completion_tokens=2048)
  text=resp.choices[0].message.content or ''
  traces.append(dict(block=len(traces),text=text,prompt_sha256=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest(),truncated=resp.choices[0].finish_reason=='length'))
  return text
 Q['get_gpt_response']=response
 async def invoke():return await Q['process_row'](copy.deepcopy(row),asyncio.Semaphore(1))
 entry=asyncio.run(invoke());v=metrics([entry])
 return dict(id=row['id'],traces=traces,planner_entry=entry,error=None,element_acc=int(v['element_acc']),operation_f1=float(v['operation_f1']),correct=bool(v['step_acc']))
def verify(row,p):
 e=p['planner_entry']
 for k,v in row.items():assert e[k]==v
 assert e['response']==p['traces'][-1]['text'] and e['ans_block']==p['traces'][-1]['block']
 v=metrics([e]);assert p['element_acc']==v['element_acc'] and p['operation_f1']==v['operation_f1'] and p['correct']==bool(v['step_acc'])
def aggregate(rows,preds):
 def group(rr):
  pp=[preds[r['id']] for r in rr];v=metrics([p['planner_entry'] for p in pp]);return dict(total=len(pp),correct=sum(p['correct'] for p in pp),step_success_rate=float(v['step_acc']),element_accuracy=float(v['element_acc']),operation_f1=float(v['operation_f1']),errors=sum(p['error'] is not None for p in pp),official_metrics={k:float(x) for k,x in v.items()})
 out={'overall':group(rows)}
 for s in ['task','website','domain']:
  rr=[r for r in rows if r['split']==s];v=group(rr);v['by_website']={k:group([r for r in rr if r['website']==k]) for k in sorted({r['website'] for r in rr})};out['cross_'+s]=v
 return out
