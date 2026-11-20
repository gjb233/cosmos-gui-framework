from pathlib import Path
import os,sys,json,time,hashlib,base64,io,argparse,ast
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from safetensors.torch import load_file
from peft import LoraConfig,get_peft_model,set_peft_model_state_dict

arg=argparse.ArgumentParser();arg.add_argument('--step',type=int,required=True);arg.add_argument('--shard',type=int,required=True);arg.add_argument('--shards',type=int,required=True);arg.add_argument('--limit',type=int,default=0)
args=arg.parse_args()
ROOT=Path('/data/mai/evaluation/ckpts-through-200-20261113');ASSET=Path('/data/mai/evaluation/assets-20261113')
MODEL='/data/mai/benchmarks/gb256-20261112/assets/mai-base/model'
TRAIN=Path('/data/mai/training/mai-ui-zero2-mlp-strict10k-20261113/run')
out=(ROOT/'smoke' if args.limit else ROOT)/f'step-{args.step:06d}';out.mkdir(parents=True,exist_ok=True)
torch.manual_seed(42);torch.set_num_threads(2)
model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'})
audit=dict(step=args.step,shard=args.shard,engine='transformers',dtype='float16',seed=42,temperature=0,max_new_tokens=512,image_resize='original resolution nearest32; min1024 max16777216',adapter_tensors=0)
if args.step:
 path=TRAIN/'ar-adapters'/f'ar-step-{args.step:06d}.safetensors'
 state=load_file(str(path));assert len(state)==504
 config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM')
 model=get_peft_model(model,config)
 mapped={k.replace('language_model.model.','base_model.model.model.language_model.'):v for k,v in state.items()}
 expected={k.replace('.default.','.') for k,p in model.named_parameters() if 'lora_' in k}
 assert set(mapped)==expected,(list(set(mapped)-expected)[:5],list(expected-set(mapped))[:5])
 result=set_peft_model_state_dict(model,mapped)
 assert not result.unexpected_keys and not any('lora_' in k for k in result.missing_keys),result
 actual={k.replace('.default.','.'):p for k,p in model.named_parameters() if 'lora_' in k}
 for k,v in mapped.items():assert torch.equal(actual[k].detach().cpu(),v.to(actual[k].dtype)),k
 audit.update(adapter_tensors=len(state),adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 model=model.merge_and_unload()
model.eval();processor=AutoProcessor.from_pretrained(MODEL)
(out/f'load-audit-{args.shard}.json').write_text(json.dumps(audit,indent=2))
def generate(messages):
 images=[];converted=[]
 for msg in messages:
  content=msg['content']
  if isinstance(content,list):
   converted_content=[]
   for part in content:
    if part['type']=='image_url':
     raw=base64.b64decode(part['image_url']['url'].split(',',1)[1]);images.append(Image.open(io.BytesIO(raw)).convert('RGB'));converted_content.append({'type':'image'})
    else:converted_content.append(part)
   content=converted_content
  converted.append(dict(role=msg['role'],content=content))
 assert len(images)==1
 prompt=processor.apply_chat_template(converted,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 inputs=processor(text=[prompt],images=images,return_tensors='pt',images_kwargs={'size':{'shortest_edge':1024,'longest_edge':16777216}}).to('cuda')
 with torch.inference_mode():output=model.generate(**inputs,do_sample=False,max_new_tokens=512,use_cache=True)
 tokens=output[0,inputs['input_ids'].shape[-1]:]
 text=processor.tokenizer.decode(tokens,skip_special_tokens=True)
 return text,len(tokens),len(tokens)==512

for bench in ('grounding','mind2web'):
 d=ASSET/bench;sys.path.insert(0,str(d))
 for module in ('native','native_protocol','evaluation_protocol','protocol','official_metrics'):sys.modules.pop(module,None)
 import native
 if bench=='grounding':
  from protocol import plan_ans
 else:
  from official_metrics import metrics
  source=(d/'client.py').read_text();tree=ast.parse(source)
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score']
  import math
  ns=dict(parse=native.convert,metrics=metrics,math=math);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-score>','exec'),ns);score=ns['score']
 rows=[json.loads(l) for l in (d/'manifest.jsonl').open()]
 selected=rows[args.shard::args.shards]
 if args.limit:selected=selected[:args.limit]
 dest=out/f'{bench}-predictions-{args.shard}.jsonl'
 done={json.loads(l)['id'] for l in dest.open()} if dest.exists() else set()
 for row in selected:
  if row['id'] in done:continue
  start=time.monotonic()
  if bench=='grounding':
   text,n,truncated=generate(native.ground_messages(row));a=native.convert(text)
   plan,ans,error=plan_ans(row,json.dumps(a or {}))
   result=dict(id=row['id'],dataset=row['dataset'],response=text,output_tokens=n,truncated=truncated,plan=plan,ans=ans,point_parse_error=error,native_parse_error=a is None)
  else:
   traces=[]
   for block in range(len(row['images'])):
    text,n,truncated=generate(native.m2_messages(row,block));traces.append(dict(block=block,text=text,output_tokens=n,truncated=truncated))
    a=native.convert(text)
    if not(a and a['action_type'].lower()=='scroll' and a['value'].lower()=='down'):break
   result=dict(id=row['id'],split=row['split'],traces=traces,**score(row,text,block))
  result['seconds']=time.monotonic()-start
  with dest.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
  done.add(row['id']);print('EVALUATED',args.step,bench,args.shard,len(done),len(selected),flush=True)
  (out/f'progress-{args.shard}.json').write_text(json.dumps(dict(benchmark=bench,completed=len(done),total=len(selected),time=time.time())))
 assert len(done)==len(selected)
(out/f'complete-{args.shard}.json').write_text(json.dumps(dict(completed=True,step=args.step,shard=args.shard,shards=args.shards,smoke=bool(args.limit))))
