from pathlib import Path
import os,json,subprocess,fcntl
W=Path('/data/mai/training/gui-libra-awhigh-aclow-20261116');C=Path('/data/mai/code/f2f8996')
out=W/'run';plan=json.loads((W/'training-plan.json').read_text())
assert plan.get('ready_to_launch') is True
assert json.loads((W/'processor-validation.json').read_text()).get('passed') is True
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU occupied'
lock=(out/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
latest=out/'checkpoints/latest-complete.json';start=0
if latest.exists():
 state=json.loads(latest.read_text());assert state['plan']==plan;start=state['completed_steps']
assert start<plan['optimizer_steps'],'training already complete'
env=os.environ.copy();env.pop('MOT_CHECKPOINT_PATH',None)
env.update(PYTHONPATH=str(W)+':'+str(C)+':'+str(C/'experiments/mai_20261111/native_template'),LD_LIBRARY_PATH='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',GUI_SUPERVISION_CACHE='inline-verified-manifest',GUI_AR_ACTIVATION_CHECKPOINT='0',PYTORCH_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false',GUI_ADAPTER_EXPORT_ROOT=str(out/'audit'),GUI_ADAPTER_EXPORT_EVERY='50',GUI_RUN_NAME=W.name,IMAGINAIRE_OUTPUT_ROOT=str(out/'runtime'),GUI_START_DRAW=str(start*256))
cmd=['/data/mai/envs/mai/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(W/'train_entry.py'),'--config',str(W/'config.json'),'--manifest',plan['manifest'],'--phase1',str(W/'phase1.jsonl'),'--phase2',str(W/'phase2.jsonl'),'--native-config',str(W/'native.toml')]
p=subprocess.Popen(cmd,cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=(out/f'train-from-{start:06d}.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
(out/'active.json').write_text(json.dumps(dict(pid=p.pid,command=cmd,resumed_from=start,plan=plan)))
print('STARTED',p.pid,'FROM',start,flush=True)
rc=p.wait();(out/'exit.json').write_text(json.dumps(dict(exit_code=rc,resumed_from=start)));print('EXIT',rc,flush=True)
