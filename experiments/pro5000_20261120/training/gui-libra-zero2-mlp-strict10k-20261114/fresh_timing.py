import os,time,json,torch
import torch.distributed as dist
from pathlib import Path
from cosmos_framework.trainer import ImaginaireTrainer as T
from cosmos_framework.gui_mot.training_artifacts import NativeTrainingArtifacts as A
fetch=T._fetch_data_batch
def timed(self,model,it):
 global trainer
 trainer=self
 if getattr(self,'_fresh_start',None) is None:
  dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();self._fresh_start=time.monotonic()
 batch,stop=fetch(self,model,it)
 cap=batch.get('ai_caption')
 while isinstance(cap,list):cap=cap[0]
 if isinstance(cap,str):
  c=json.loads(cap)
  with (Path(os.environ['GUI_ADAPTER_EXPORT_ROOT'])/f'draws-rank{dist.get_rank()}.jsonl').open('a') as f:f.write(json.dumps(c)+'\n')
 return batch,stop
T._fetch_data_batch=timed
end=A.on_training_step_end
def finish(self,model,data_batch,output_batch,loss,iteration=0):
 end(self,model,data_batch,output_batch,loss,iteration=iteration)
 dist.barrier();torch.cuda.synchronize()
 row=dict(step=iteration,seconds=time.monotonic()-trainer._fresh_start,loss=float(loss),global_batch=256,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30)
 trainer._fresh_start=None
 with (Path(os.environ['GUI_ADAPTER_EXPORT_ROOT'])/f'timing-rank{dist.get_rank()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
A.on_training_step_end=finish
