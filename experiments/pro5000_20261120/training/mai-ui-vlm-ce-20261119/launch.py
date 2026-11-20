from pathlib import Path
import subprocess,os,json
w=Path(__file__).parent;r=w/'run'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
e=os.environ.copy();e.update(OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',WANDB_MODE='disabled')
cmd=['/data/mai/envs/mai/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(w/'train.py')]
p=subprocess.Popen(cmd,cwd=w,env=e,stdout=(r/'train.log').open('a'),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
(r/'active.json').write_text(json.dumps(dict(pid=p.pid,command=cmd)))
rc=p.wait();(r/'exit.json').write_text(json.dumps(dict(exit_code=rc)))
