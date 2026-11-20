"""Read-only FM sampling from full MAI CE+FM checkpoints; no optimizer."""
import os, sys, json, math, hashlib, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

OUT=Path(__file__).resolve().parent
TRAIN=OUT.parent/'mai-ui-70283-resume150-fsdp7-micro4-gb56-a22-fullmem'
SOURCE=Path('/rscratch/wzzheng/experiments/gui-grounding-fixed-ab-20261002-v1/source')
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(TRAIN.parent/'unified-template-debug'))
sys.path.insert(0,str(TRAIN))
import protocol
from patch_training import install
install()
import training_backend as backend
records={x['row']['id']:x['row'] for x in json.loads((OUT/'selection.json').read_text())}
backend.install(records,[dict(id=k,level='high') for k in records])
from cosmos_framework.gui_mot import mixed_supervision as ms,native_model as nm
# Empty target is the native sampler's explicit inference representation.
def infer_batch(batch,**kwargs):
    d=dict(batch)
    for k in ('gui_action_text','ai_caption','gui_system_prompt'):
        v=d[k]
        while isinstance(v,list) and len(v)==1 and isinstance(v[0],list):v=v[0]
        d[k]=v
    assert all(x=='' for x in d['gui_action_text'])
    assert all(x==protocol.SYSTEM for x in d['gui_system_prompt'])
    return d
ms.prepare_grouped_batch=infer_batch
from cosmos_framework.gui_mot import attention as att
from torch.nn.attention import sdpa_kernel, SDPBackend
original_sdpa=att._sdpa
def fused_sdpa(q,k,v,*,causal=False):
    # Efficient SDPA does not implement enable_gqa. Materialize the equivalent
    # repeated KV heads while preserving causal masking and all input tokens.
    if q.shape[1]!=k.shape[1]:
        assert q.shape[1]%k.shape[1]==0 and k.shape[1]==v.shape[1]
        repeat=q.shape[1]//k.shape[1]
        k=k.repeat_interleave(repeat,dim=1)
        v=v.repeat_interleave(repeat,dim=1)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):return original_sdpa(q,k,v,causal=causal)
att._sdpa=fused_sdpa
class InferenceDataset:
    def __init__(self,*args,**kwargs):self.items=json.loads((OUT/'selection.json').read_text())
    def __len__(self):return len(self.items)
    def __getitem__(self,index):
        from cosmos_framework.gui_mot.joint_dataset import make_sample,read_screen
        item=self.items[index%len(self.items)];row=item['row'];h,w=item['hw']
        with Image.open(row['current_image']) as im:orig=np.array(im.convert('RGB'))
        prefix=torch.from_numpy(orig.copy()).permute(2,0,1).contiguous()
        sample=make_sample(read_screen(row['current_image'],(h,w)),row['goal'],native_ar_text=False,video_only=True,prefix_image=prefix)
        sample['ai_caption']=item['caption'];sample['gui_system_prompt']=protocol.SYSTEM;sample['gui_action_text']=''
        return sample
ms.MixedJointPolicyDataset=InferenceDataset

from cosmos_framework.trainer import ImaginaireTrainer

def thash(t):
    x=t.detach().contiguous().cpu()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()

@torch.no_grad()
def infer(self,model,*unused):
    rank=dist.get_rank();step=int(os.environ['FM_STEP'])
    out=OUT/f'step-{step:06d}';out.mkdir(exist_ok=True)
    model=model.to('cuda',memory_format=self.config.trainer.memory_format)
    model.on_train_start(self.config.trainer.memory_format)
    self.checkpointer.load(model)
    model.eval()
    vae_compute='bfloat16'
    # Reconstruct HF nonpersistent rotary buffer just as in the training patch.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding
    visual=model.net.language_model.visual
    rotary=Qwen3VLVisionRotaryEmbedding((visual.config.hidden_size//visual.config.num_heads)//2).inv_freq
    visual.rotary_pos_emb.register_buffer('inv_freq',rotary.to(next(visual.parameters()).device),persistent=False)
    from safetensors.torch import load_file
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    expected=load_file(str(OUT/f'weights-{step}.safetensors'))
    restored=set()
    for n,param in model.net.named_parameters():
        name=n.replace('_checkpoint_wrapped_module.','')
        if name not in expected:continue
        value=expected[name]
        assert value.shape==param.shape
        if isinstance(param,DTensor):
            shape,offset=compute_local_shape_and_global_offset(param.shape,param.device_mesh,param.placements)
            part=value[tuple(slice(i,i+n) for i,n in zip(offset,shape))]
            param.to_local().copy_(part.to(param.device,param.dtype))
            assert torch.equal(param.to_local().cpu(),part.to(param.dtype))
        else:
            param.copy_(value.to(param.device,param.dtype))
            assert torch.equal(param.cpu(),value.to(param.dtype))
        restored.add(name)
    assert restored==set(expected) and len(restored)==584
    model._resume_draw_checked=True
    if rank==0:(out/'loading-audit.json').write_text(json.dumps(dict(step=step,restored_parameters=584,ar_lora=288,fm_lora=288,optimizer_steps=0,base_checkpoint=str(self.config.checkpoint.load_path))))
    del expected
    from cosmos_framework.gui_mot.joint_dataset import make_sample,read_screen
    from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
    items=json.loads((OUT/'selection.json').read_text())
    # One GPU, fixed per-sample seeds.
    for item in items:
        started=time.time();row=item['row'];h,w=item['hw']
        with Image.open(row['current_image']) as im:orig=np.array(im.convert('RGB'))
        assert hashlib.sha256(orig.tobytes()).hexdigest()==item['current_rgb_sha256']
        prefix=torch.from_numpy(orig.copy()).permute(2,0,1).contiguous()
        current=read_screen(row['current_image'],(h,w))
        sample=make_sample(current,row['views']['high']['instruction'],native_ar_text=False,video_only=True,prefix_image=prefix)
        sample['ai_caption']=item['caption'];sample['gui_system_prompt']=protocol.SYSTEM
        sample['gui_action_text']=''
        assert torch.count_nonzero(sample['video'][:,1:]).item()==0
        batch=custom_collate_fn([sample])
        # No ground-truth action or future image enters model sampling.
        result=model.generate_samples_from_batch(batch,seed=[item['seed']],num_steps=20,guidance=1.0,use_batched_cfg=False,upsample_task=None)
        assert model._gui_action_target_ids is None
        z=result['vision'][0];z=z.unsqueeze(0) if z.ndim==4 else z
        assert torch.isfinite(z).all()
        decoded=model.decode(z)
        assert torch.isfinite(decoded).all()
        pixels=((decoded[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
        assert pixels.shape[1:3]==(h,w) and len(pixels)==5
        if rank==0:
            d=out/row['id'].replace(':','-');d.mkdir(exist_ok=True)
            for j,frame in enumerate(pixels):Image.fromarray(frame).save(d/f'frame-{j}.png')
            torch.save(z.cpu(),d/'latent.pt')
            meta=dict(id=row['id'],step=step,seed=item['seed'],sampling_steps=20,guidance=1.,future_target_supplied=False,current_gold_action_supplied=False,caption_sha256=hashlib.sha256(item['caption'].encode()).hexdigest(),latent_sha256=thash(z),latent_shape=list(z.shape),seconds=time.time()-started,display_frame=4)
            (d/'metadata.json').write_text(json.dumps(meta,indent=2))
            print('FM_SAMPLE',json.dumps(meta),flush=True)
            (OUT/'status.json').write_text(json.dumps(dict(stage='sampling',step=step,id=row['id'],time=time.time())))
        dist.barrier()
    if rank==0:(out/'complete.json').write_text(json.dumps(dict(step=step,samples=len(items),passed=True,optimizer_steps=0)))
    dist.barrier();dist.destroy_process_group()

ImaginaireTrainer.train=infer
from cosmos_framework.gui_mot.train_joint import main
main()
