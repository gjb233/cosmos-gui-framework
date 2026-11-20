"""Native FM pack of eight samples with isolated native AR forwards."""
import copy,json,os
from collections import deque
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from cosmos_framework.gui_mot import multimodal as mm
from cosmos_framework.gui_mot.resumable_loader import GuiPackingDataLoader

def reset_fresh(self,iteration):
 assert iteration==0 and self.max_samples_per_batch==8
 inner=self.dataloader_list[0]
 assert inner.dataloader.num_workers==0
 inner.dataset.set_start_sample(0)
 self.dataloaders=[iter(inner)];self.buffers=[deque()];self.global_id=0
GuiPackingDataLoader.set_start_iteration=reset_fresh

single=mm.hybrid_ar_forward
def batched_ar(network,packed_seq,last_hidden_state,output_dict):
 assert packed_seq.gui_native_fm_ar and packed_seq.gui_mot_joint
 count=len(packed_seq.gui_action_target_ids)
 assert count==len(packed_seq.gui_multimodal_inputs)==8
 outputs=[]
 for i in range(count):
  item=copy.copy(packed_seq)
  for name in ('gui_action_target_ids','gui_multimodal_inputs','gui_action_target_masks','gui_action_target_weights'):
   if hasattr(packed_seq,name):setattr(item,name,[getattr(packed_seq,name)[i]])
  row={};single(network,item,last_hidden_state,row);outputs.append(row)
 length=max(row['gui_ar_labels'].shape[1] for row in outputs)
 for key in ('gui_ar_logits','gui_ar_labels','gui_ar_token_mask','gui_ar_token_weights'):
  values=[]
  for row in outputs:
   value=row[key];pad=length-value.shape[1]
   values.append(F.pad(value,(0,0,0,pad) if key=='gui_ar_logits' else (0,pad),value=0))
  output_dict[key]=torch.cat(values,dim=0)
 assert output_dict['gui_ar_logits'].shape[:2]==output_dict['gui_ar_labels'].shape
 assert torch.allclose(output_dict['gui_ar_token_weights'].sum(-1),torch.ones(8,device=last_hidden_state.device))
mm.hybrid_ar_forward=batched_ar

def audit_batch(model,batch,step,micro):
 captions=batch['ai_caption']
 while isinstance(captions,list) and len(captions)==1 and isinstance(captions[0],list):captions=captions[0]
 assert isinstance(captions,list) and len(captions)==8
 captions=[json.loads(x) for x in captions]
 expected=[dist.get_rank()+8*((step*4+micro)*8+i) for i in range(8)]
 assert [c['draw_slot'] for c in captions]==expected
 assert all(c['slot_weight']==1. for c in captions)
 root=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT'])
 with (root/f'draws-rank{dist.get_rank()}.jsonl').open('a') as f:
  f.write(json.dumps(dict(step=step+1,micro=micro,samples=8,draws=captions))+'\n')

def audit_outputs(model,step,micro):
 assert len(model._gui_prefix)==len(model._gui_action_target_ids)==8
 if step==0 and micro==0:
  import protocol
  for prefix,caption in zip(model._gui_prefix,model._history_fm_contexts,strict=True):
   protocol.assert_decoded_prefix(model._gui_processor(),prefix,json.dumps(caption))
  p=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT'])/f'actual-batch-rank{dist.get_rank()}.json'
  p.write_text(json.dumps(dict(samples=8,prefix_tokens=[x['input_ids'].shape[-1] for x in model._gui_prefix],target_tokens=[len(x) for x in model._gui_action_target_ids],ar='independent causal forwards, merged masked CE',fm='native eight-sample sequence packing')))
