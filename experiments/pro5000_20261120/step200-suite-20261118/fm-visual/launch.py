from pathlib import Path
import subprocess,os,json
V=Path(__file__).parent;W=Path('/data/mai/training/gui-libra-awhigh-aclow-fm2-20261117');R=Path('/data/mai/evaluation/step200-suite-20261118/fm-visual')
env=os.environ.copy();env.update(PYTHONPATH=str(V)+':'+str(W)+':/data/mai/code/f2f8996:/data/mai/code/f2f8996/experiments/mai_20261111/native_template',LD_LIBRARY_PATH='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',GUI_SUPERVISION_CACHE='inline-verified-manifest',GUI_START_DRAW='0',GUI_ADAPTER_EXPORT_ROOT=str(R/'audit'),GUI_RUN_NAME='fm-visual-step200',PYTORCH_ALLOC_CONF='expandable_segments:True',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',IMAGINAIRE_OUTPUT_ROOT=str(R/'runtime'))
(R/'audit').mkdir(exist_ok=True)
cmd=['/data/mai/envs/mai/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(V/'entry.py'),'--config',str(W/'config.json'),'--manifest',str(W/'train-manifest.jsonl'),'--phase1',str(W/'phase1.jsonl'),'--phase2',str(W/'phase2.jsonl'),'--native-config',str(W/'native.toml')]
subprocess.run(cmd,cwd='/data/mai/code/f2f8996',env=env,check=True)
