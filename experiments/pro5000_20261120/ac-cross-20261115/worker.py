from pathlib import Path
import os,sys,json,time,argparse,logging,ast
OFF=Path('/data/mai/benchmarks/gui-libra-official-repro');sys.path.insert(0,str(OFF))
from android_world import registry,suite_utils
from android_world.agents import seeact_v_guilibra,infer
from android_world.env import env_launcher
from android_env.wrappers import a11y_grpc_wrapper
a11y_grpc_wrapper._get_accessibility_forwarder_apk=lambda:Path('/data/mai/benchmarks/androidworld/a11y-cached.apk').read_bytes()
p=argparse.ArgumentParser();p.add_argument('--step',type=int);p.add_argument('--port',type=int);p.add_argument('--console-port',type=int);p.add_argument('--grpc-port',type=int);p.add_argument('--slot',type=int);p.add_argument('--slots',type=int);args=p.parse_args()
R=Path('/data/mai/evaluation/ac-cross-20261115/androidworld')/('round-'+os.environ['LIBRA_REPRO_ROUND'])
root=R/'full/step-000000'/f'slot-{args.slot}';root.mkdir(parents=True,exist_ok=True)
log=logging.getLogger('official-repro');log.setLevel(logging.INFO);log.addHandler(logging.FileHandler(root/'official.log'))
env=env_launcher.load_and_setup_env(adb_path='/data/mai/benchmarks/androidworld/sdk/platform-tools/adb',console_port=args.console_port,grpc_port=args.grpc_port,emulator_setup=True)
try:
 reg=registry.TaskRegistry();suite=suite_utils.create_suite(reg.get_registry(reg.ANDROID_WORLD_FAMILY),n_task_combinations=1,seed=42);names=list(suite);selected=[(i,n) for i,n in enumerate(names) if n!='SimpleSmsReplyMostRecent'][args.slot::args.slots]
 (root/'tasks.json').write_text(json.dumps(dict(all_tasks=names,selected=[n for _,n in selected],seed=42,excluded='SimpleSmsReplyMostRecent')))
 class Client:
  def reset(self,go_home=False):return env.reset(go_home=go_home)
  def get_screenshot(self,wait_to_stabilize=False):return env.get_state(wait_to_stabilize=wait_to_stabilize).pixels.copy()
  def execute_action(self,action):return env.execute_action(action)
  def get_task_template(self,task_type,task_idx):return suite[task_type][task_idx].template
  def get_task_goal(self,task_type,task_idx):return suite[task_type][task_idx].goal
  def initialize_task(self,task_type,task_idx):return suite[task_type][task_idx].initialize_task(env)
  def tear_down_task(self,task_type,task_idx):return suite[task_type][task_idx].tear_down(env)
  def get_task_score(self,task_type,task_idx):return suite[task_type][task_idx].is_successful(env)
 client=Client();llm=infer.Gpt4Wrapper('GUI-Libra/GUI-Libra-8B',temperature=0.0,base_url=f'http://127.0.0.1:{args.port}/v1')
 from prompt_adapter import wrap_aw
 llm=wrap_aw(llm,root)
 agent=seeact_v_guilibra.SeeAct_V(client,llm,file_logger=log,save_img_dir=str(root/'images'),no_guidance=True)
 source=(OFF/'run_suite_on_docker_mp.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='eval_one_task')
 ns=dict(time=time,logging=logging,logger=log,AndroidEnvClient=Client,Tuple=tuple);exec(compile(ast.Module(body=[node],type_ignores=[]),str(OFF/'run_suite_on_docker_mp.py'),'exec'),ns)
 rows=[]
 for index,name in selected:
  ok,score,msg=ns['eval_one_task'](client,agent,name,index,0,20,log)
  last=agent.history[-1].get('action_output_json') if agent.history else None
  ended_by=getattr(last,'action_type',None)
  row=dict(task=name,success=bool(ok),score=float(score),termination_action=ended_by,paper_terminate_success=bool(ok and ended_by=='status'),environment_error=msg.startswith('Exception'),steps=len(agent.history))
  (root/f'trace-{index}.json').write_text(json.dumps([{k:h.get(k) for k in ['action_prompt','action_output','action','summary']} for h in agent.history],ensure_ascii=False));rows.append(row);(root/'results.json').write_text(json.dumps(rows,indent=2));print('TASK_COMPLETED',name,ok,flush=True)
 (root/'complete.json').write_text(json.dumps(dict(completed=True,tasks=len(rows))))
finally:env.close()
