import argparse,base64,io,json,sys,time,urllib.request,os
from pathlib import Path
from PIL import Image
from android_world import registry,suite_utils,checkpointer
from android_world.agents.base_agent import EnvironmentInteractingAgent,AgentInteractionResult
from android_world.env import env_launcher,json_action
from android_env.wrappers import a11y_grpc_wrapper
a11y_grpc_wrapper._get_accessibility_forwarder_apk=lambda: Path('/data/mai/benchmarks/androidworld/a11y-cached.apk').read_bytes()
sys.path.insert(0,'/data/mai/libra-fm2-eval50-150-20261118/aw50')
from libra_protocol import SYSTEM,parse,messages as libra_messages,android_action
p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);p.add_argument('--port',type=int,default=25600);p.add_argument('--console-port',type=int,default=5554);p.add_argument('--grpc-port',type=int,default=8554);p.add_argument('--smoke',action='store_true');p.add_argument('--slot',type=int,default=0);p.add_argument('--slots',type=int,default=8);args=p.parse_args()
root=Path('/data/mai/evaluation/libra-fm2-eval50-150-20261118/aw50')/('smoke' if args.smoke else 'full')/f'step-{args.step:06d}'/f'slot-{args.slot}';root.mkdir(parents=True,exist_ok=True)
class Agent(EnvironmentInteractingAgent):
 def reset(self,go_home=False):
  self.history=[];super().reset(go_home)
 def step(self,goal):
  state=self.get_post_transition_state();im=Image.fromarray(state.pixels).convert('RGB');b=io.BytesIO();im.save(b,format='PNG')
  messages=libra_messages(im,goal,self.history)
  body=json.dumps(dict(model='eval',messages=messages,max_tokens=2048,temperature=0,seed=42)).encode()
  request=urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',body,{'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=180) as response:result=json.load(response)
  raw=result['choices'][0]['message']['content'];data=dict(input_text=[m for m in messages if m['role']=='system']+[{'role':'user','content':[x for x in messages[1]['content'] if x['type']=='text']}],raw_response=raw,screenshot=state.pixels,goal=goal,action=None,action_error=None,finish_reason=result['choices'][0]['finish_reason'],thinking_present='<thinking>' in raw);done=False
  try:
   obj=parse(raw);kw,done=android_action(obj,im.width,im.height)
   
   if obj['action_type'].lower() in ('write','type_text') and obj['point_2d']!=[-100,-100]:
    p=obj['point_2d'];self.env.execute_action(json_action.JSONAction(action_type='click',x=round(p[0]*(im.width-1)/1000),y=round(p[1]*(im.height-1)/1000)))
   action=json_action.JSONAction(**kw);self.env.execute_action(action);self.history.append(obj);data['action']=kw
  except (ValueError,AssertionError,KeyError,TypeError) as e:data['action_error']=repr(e)
  with (root/'interactions.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in data.items() if k!='screenshot'},ensure_ascii=False)+'\n')
  return AgentInteractionResult(done,data)
env=env_launcher.load_and_setup_env(adb_path=f'/data/mai/libra-fm2-eval50-150-20261118/aw50/adb-{args.slot}',console_port=args.console_port,grpc_port=args.grpc_port,emulator_setup=os.environ.get('AW_REBUILD_SNAPSHOTS')=='1')
try:
 r=registry.TaskRegistry();suite=suite_utils.create_suite(r.get_registry(r.ANDROID_WORLD_FAMILY),n_task_combinations=1,seed=30,tasks=['ContactsAddContact'] if args.smoke else None);suite.suite_family=r.ANDROID_WORLD_FAMILY
 all_names=list(suite)
 selected=all_names[args.slot::args.slots] if not args.smoke else all_names
 for name in list(suite):
  if name not in selected:del suite[name]
 (root/'tasks.json').write_text(json.dumps(dict(seed=30,all_tasks=all_names,selected=selected)))
 agent=Agent(env,name=f'GUI-Libra-step{args.step}',transition_pause=None)
 results=suite_utils.run(suite,agent,checkpointer=checkpointer.IncrementalCheckpointer(str(root/'episodes')),demo_mode=False)
 (root/'results.json').write_text(json.dumps(results,default=str,indent=2))
 (root/'complete.json').write_text(json.dumps(dict(completed=True,step=args.step,seed=30,templates=len(selected))))
finally:env.close()
