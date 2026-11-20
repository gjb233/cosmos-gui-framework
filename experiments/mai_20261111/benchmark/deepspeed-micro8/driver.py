"""Eight real MAI microbatches, decoder outputs and recorded CE/FM adjoints."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe, Schedule1F1B

root = Path('/rscratch/wzzheng/experiments/mixed-sft-preparation-20261028-v1/mai-framework-comparison-a22-seven-20261107')
work = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "torch-pipeline"))
from flat_pipeline_stage import FlatJointPipelineStage
from official_schedule_adapter import OrderedJointStage


def load(path, device):
    return torch.load(path, map_location=device, weights_only=False)

def run(kind):
    import deepspeed
    from deepspeed.pipe import PipelineModule
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 7
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    ref = root / "real-stage-reference/reference"
    from joint_stage import JointDecoderStage
    starts=(0,3,9,15,21,26,31,36)
    cuts=starts
    lo,hi=cuts[rank:rank+2]
    layers=[]
    for j,(a,b) in enumerate(zip(starts,starts[1:])):
        if lo<b and hi>a:
            original=load(ref / f"stage-{j}.pt","cpu")
            assert original.start==a and len(original.layers)==b-a
            layers.extend(original.layers[max(0,lo-a):min(b,hi)-a])
            del original
    assert len(layers)==hi-lo
    stage=JointDecoderStage(layers,0,len(layers))
    stage.start=lo
    # RMSNorm is independent along the sequence axis; bound its FP32 temporary.
    for module in stage.modules():
        if type(module).__name__ == "Qwen3VLTextRMSNorm":
            original_norm = module.forward
            def bounded_norm(x, original=original_norm):
                shape=x.shape
                x=x.reshape(-1,shape[-1])
                if x.shape[0] <= 512:
                    return original(x).reshape(shape)
                return torch.cat([original(part) for part in x.split(512,dim=0)],dim=0).reshape(shape)
            module.forward = bounded_norm

    for p in stage.parameters():
        p.data = p.data.to(device, torch.float32 if p.requires_grad else torch.bfloat16)
    for m in stage.modules():
        for name, buf in m.named_buffers(recurse=False):
            m._buffers[name] = buf.to(device)
    def read(k):
        return load(ref / f"input-{k}.pt", device)
    caps=(1,1,1)
    dtypes=(torch.bfloat16,)*3
    from torch.utils.checkpoint import checkpoint
    def chunked_forward(original):
        def forward(x):
            if x.shape[-2] <= 256:
                return original(x)
            return torch.cat([checkpoint(original,part,use_reentrant=False) for part in x.split(256,dim=-2)],dim=-2)
        return forward
    bounded=0
    for module in stage.modules():
        if type(module).__name__=='Qwen3VLTextMLP':
            module.forward=chunked_forward(module.forward)
            bounded+=1
    assert bounded>0
    print('token-chunked MLP modules',rank,bounded,flush=True)
    parts = [JointDecoderStage([layer],0,1) for layer in stage.layers]
    for i,part in enumerate(parts):
        part.start=lo+i
    def recomputed(fm_pack, fm_position_embeddings, ar_hidden, ar_cos, ar_sin, ar_visual_mask, ar_deepstack, fm_visual=None, attention_mask=None):
        for part in parts:
            fm_pack,ar_hidden=checkpoint(part,fm_pack,fm_position_embeddings,ar_hidden,ar_cos,ar_sin,ar_visual_mask,ar_deepstack,fm_visual=fm_visual,attention_mask=attention_mask,use_reentrant=False,preserve_rng_state=True)
        return fm_pack,ar_hidden
    stage.forward=recomputed
    flat = FlatJointPipelineStage(stage, caps)
    ordered = OrderedJointStage(flat)
    import cloudpickle
    from types import SimpleNamespace
    from native_endpoints import NativeEndpoints
    from metadata_channel import MicrobatchMetadataChannel
    from cosmos_framework.gui_mot.native_model import GuiJointModel
    import training_backend
    control = dist.new_group(backend="gloo")
    last = dist.new_group(ranks=[6],backend="nccl")
    channel = MicrobatchMetadataChannel(control,dist.group.WORLD,device)
    endpoint_dir=root/"native-endpoint-reference/endpoint"
    ep=None
    if rank in (0,6):
        training_backend.install({},[])
        with (root/"endpoint-constructor/constructor.pkl").open("rb") as f: c=cloudpickle.load(f)
        net=c["net"]
        net.load_state_dict(load(endpoint_dir/"weights.pt","cpu"),strict=True,assign=True)
        if rank==0:
            net.language_model.lm_head=torch.nn.Identity()
            net.llm2vae=torch.nn.Identity()
        if rank==6:
            # Heads consume decoded hidden states, never pixels or token IDs.
            net.language_model.visual = torch.nn.Identity()
            net.language_model.model.embed_tokens = torch.nn.Identity()
        for n,p in net.named_parameters():
            owned=(n.startswith(("time_embedder.","vae2llm.")) if rank==0 else n.startswith("llm2vae."))
            p.requires_grad_(p.requires_grad and owned)
            p.data=p.data.to(device,torch.float32 if p.requires_grad else torch.bfloat16)
        if rank==0:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding
            visual=net.language_model.visual
            freq=Qwen3VLVisionRotaryEmbedding((visual.config.hidden_size//visual.config.num_heads)//2).inv_freq
            before=torch.equal(visual.rotary_pos_emb.inv_freq.cpu(),freq)
            visual.rotary_pos_emb.register_buffer("inv_freq",freq,persistent=False)
        net.to(device)
        if rank==0:
            assert torch.equal(visual.rotary_pos_emb.inv_freq.cpu(),freq) and freq.dtype==torch.float32
            (work/f"rotary{rank}.json").write_text(json.dumps(dict(before=before,after=True)))
        ep=NativeEndpoints(net)
        model=SimpleNamespace(**c["objective_attributes"])
        model._loss_averaging_group=lambda:(last,1)
    local_inputs={}
    head={}
    objectives={}
    rows={}
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder=ordered
            if ep is not None:self.endpoint=net
        def forward(self,*value):
            outputs=[]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                for row in range(8):
                    key=ordered.keys[ordered.cursor]
                    sample=local_inputs.pop(key) if rank==0 else tuple(v[row] for v in value)
                    outputs.append(ordered(*sample))
            return tuple(torch.stack(parts) for parts in zip(*outputs))
    def sample_loss(wires,target):
        k=int(target.item())
        template,_,env=flat.contexts[k]
        ca,fu,ar=env.unpack(wires)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out=ep.finish_local_heads(dict(template,causal_seq=ca,full_only_seq=fu),ar,head[k])
            tgt=objectives[k]
            args=dict(tgt["arguments"]);extra=args.pop("kwargs",{})
            model._mixed_fm_mask=tgt["fm_mask"]
            loss,metrics=GuiJointModel._compute_losses(model,out,**args,**extra)
            rows[k]=[loss.item(),metrics["gui_ar_ce"].item(),metrics["flow_matching_loss_vision"].item(),
                     k]
            return loss if k<210962 else loss*0
    def native_loss(wires,target):
        assert target.numel()==8 and all(v.shape[0]==8 for v in wires)
        return torch.stack([sample_loss(tuple(v[row] for v in wires),k) for row,k in enumerate(target)]).sum()
    block=Block()
    block.train()
    params=[p for p in block.parameters() if p.requires_grad]
    accum={}
    def observe(n):
        def hook(g):
            assert g.dtype==torch.float32 and torch.isfinite(g).all(),n
            if n not in accum:accum[n]=g.detach().clone()
            else:accum[n].add_(g.detach())
        return hook
    for n,p in block.named_parameters():
        if p.requires_grad:p.register_hook(observe(n))

    class DSBlock(torch.nn.Module):
        def __init__(self):
            super().__init__();self.block=block
        def forward(self,inputs):
            return self.block(*(inputs if isinstance(inputs,tuple) else (inputs,)))
    # DeepSpeed divides each microbatch loss by GAS=7; compensate because
    # external accumulation below normalizes once by 56 effective samples.
    pipe=PipelineModule(layers=[DSBlock() if i==rank else torch.nn.Identity() for i in range(7)],num_stages=7,partition_method='uniform',loss_fn=lambda outputs,target:native_loss(outputs,target)*7,activation_checkpoint_interval=0,dynamic_shape=True)
    dsengine,_,_,_=deepspeed.initialize(model=pipe,optimizer=torch.optim.SGD(params,lr=0.),config=dict(train_batch_size=56,train_micro_batch_size_per_gpu=8,gradient_accumulation_steps=7,zero_optimization=dict(stage=0),pipeline=dict(pipe_partitioned=False,grad_partitioned=False),wall_clock_breakdown=False,steps_per_print=100000))

    import os,math,hashlib
    root_out=work;stream=root_out/'stream'
    migration=load(work/'migration-350.pt','cpu')
    def canonical(n):
        if n.startswith('endpoint.'):return n.removeprefix('endpoint.')
        i,tail=n.split('decoder.flat_stage.decoder.layers.')[1].split('.',1)
        return f'language_model.model.layers.{lo+int(i)}.{tail}'
    named={canonical(n):p for n,p in block.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.99),eps=1e-8,weight_decay=0.,fused=True)
    for name,p in named.items():
        value=migration['params'][name];assert p.shape==value.shape
        with torch.no_grad():p.copy_(value.to(device,p.dtype))
        assert torch.equal(p.cpu(),value.to(p.dtype))
        opt.state[p]={k:v.to(device) for k,v in migration['optimizer'][name].items()}
        assert int(opt.state[p]['step'])==350
    allnames=[None]*7;dist.all_gather_object(allnames,list(named),group=control)
    assert sum(map(len,allnames))==584 and set(n for ns in allnames for n in ns)==set(migration['params'])
    group_values=[migration['groups'][n] for n in named]
    assert all(v==group_values[0] for v in group_values)
    opt.param_groups[0].update(group_values[0])
    frozen_checked=0
    for i,layer in enumerate(stage.layers):
        for n,p in layer.named_parameters():
            name=f'language_model.model.layers.{lo+i}.{n}'
            if name in migration['frozen_probe_hashes']:
                digest=hashlib.sha256(p.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                assert digest==migration['frozen_probe_hashes'][name],name
                frozen_checked+=1
    assert frozen_checked==2*(hi-lo)
    torch.set_rng_state(migration['trainer'][f'rng_state_{rank}.torch'])
    torch.cuda.set_rng_state(migration['trainer'][f'rng_state_{rank}.torch_cuda'],device)
    source_scheduler=migration['scheduler']
    del migration
    (work/f'restored-rank{rank}.json').write_text(json.dumps(dict(step=350,next_draw=20800,parameters=len(named),layers=[lo,hi],frozen_probe_count=frozen_checked,optimizer_steps=350)))
    print('RESTORED_350',rank,len(named),lo,hi,flush=True)
    deadline=time.monotonic()+1800
    while not (stream/'ready').exists():
        assert time.monotonic()<deadline and not (stream/'abort').exists()
        time.sleep(.2)
    total_draws=210962
    end_step=354
    initial={n:p.detach().cpu().clone() for n,p in named.items()}
    for step in range(351,end_step+1):
        dist.barrier();torch.cuda.synchronize();start=time.monotonic()
        torch.cuda.reset_peak_memory_stats();prep_seconds=0.;schedule_seconds=0.
        draw_start=9600+(step-151)*56
        for phase in range(1):
            wave=(step-351)*1+phase
            if rank==0:(stream/f'request-{wave}').touch()
            base=draw_start+phase*56
            keys=[base+r+7*j for r in range(7) for j in range(8)]
            assert sorted(keys)==list(range(base,base+56))
            pending=[];prep_start=time.monotonic()
            if rank==0:print('WAVE_BEGIN',step,phase,base,base+55,flush=True)
            for k in keys:
                if rank in (0,6):
                    path=stream/f'{min(k,total_draws-1)}.pt'
                    deadline=time.monotonic()+1200
                    while not path.exists():
                        if time.monotonic()>deadline or (stream/'abort').exists():
                            raise RuntimeError(f'missing native draw {k}')
                        time.sleep(.03)
                    item=load(path,device)
                x=None
                if rank==0:
                    raw=item['raw']
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        prep=ep.prepare(k,*raw['args'],**raw['kwargs'])
                    pack=dict(prep['fm_pack'])
                    ca=pack.pop('causal_seq');fu=pack.pop('full_only_seq');ar=prep['ar_hidden']
                    ctx,act=ep.head_context(k)
                    x=dict(meta={n:v for n,v in prep.items() if n not in ('fm_pack','ar_hidden')},
                           pack=pack,shapes=[tuple(v.shape) for v in (ca,fu,ar)],
                           context={n:v for n,v in ctx.items() if n!='packed_seq'},act=act)
                info=channel.exchange(k,x)
                pending.append((k,info,(ca,fu,ar) if rank==0 else None,item['raw']['kwargs']['packed_seq'] if rank==6 else None,item['target'] if rank==6 else None))
            capacities=tuple(max(1,max(math.prod(x[1]['shapes'][i]) for x in pending)) for i in range(3))
            flat.capacities=capacities
            if rank==0:
                with (work/'wave-shapes.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=step,phase=phase,draws=keys,capacities=capacities,shapes=[x[1]['shapes'] for x in pending]))+'\n')
                print('WAVE_PREPARED',step,phase,capacities,flush=True)
            for k,info,values,packed_seq,tgt in pending:
                shapes=info['shapes']
                dummy=[torch.empty(1,dtype=dt,device=device).expand(sh) for dt,sh in zip(dtypes,shapes)]
                pack=dict(info['pack'],causal_seq=dummy[0],full_only_seq=dummy[1])
                env=flat.register_microbatch(k,fm_pack=pack,ar_shape=shapes[2],ar_dtype=dtypes[2],**info['meta'])
                if rank==0:local_inputs[k]=env.pack(values)
                if rank==6:
                    head[k]=(dict(info['context'],packed_seq=packed_seq),info['act']);objectives[k]=tgt
            pending.clear();ordered.begin_step(keys)
            wires=tuple(torch.empty((8,size),dtype=dt,device=device) for size,dt in zip(capacities,dtypes))
            batch=torch.ones(56,device=device)
            dsengine.reset_activation_shape()
            batches=iter((torch.ones(8,device=device),torch.tensor(keys[i:i+8],device=device)) for i in range(0,len(keys),8))
            torch.cuda.synchronize();prep_seconds+=time.monotonic()-prep_start;sched_start=time.monotonic()
            dsengine.train_batch(data_iter=batches)
            torch.cuda.synchronize();schedule_seconds+=time.monotonic()-sched_start
            for param in params:param.grad=None
            del wires
            ordered.finish_step()
            if rank==0:
                for k in keys:ep.release_after_backward(k)
            head.clear();objectives.clear();channel.clear_completed_step()
            dist.barrier()
            if rank==0:
                for k in keys:
                    if k<total_draws:(stream/f'{k}.pt').unlink()
                (stream/f'request-{wave}').unlink()
        next_draw=min(draw_start+56,total_draws);size=next_draw-draw_start
        assert len(accum)==len(params)
        ss=torch.zeros((),device=device,dtype=torch.float64)
        for n,p in block.named_parameters():
            if p.requires_grad:
                p.grad=accum.pop(n).div_(size)
                assert torch.isfinite(p.grad).all(),n
                ss+=p.grad.norm(dtype=torch.float64).square()
        dist.all_reduce(ss);norm=ss.sqrt();assert torch.isfinite(norm)
        scale=(1./(norm+1e-6)).clamp(max=1.).float()
        for p in params:p.grad.mul_(scale)
        equivalent=150+min((step-150)*56,total_draws-9600)/64
        opt.param_groups[0]['lr']=1e-5*(3298-equivalent)/(3298-33)
        opt.step();assert all(int(v['step'])==step for v in opt.state.values())
        opt.zero_grad(set_to_none=True)
        dist.barrier();torch.cuda.synchronize();seconds=time.monotonic()-start
        changed=dict(ar=0,fm=0,other=0)
        if step==351:
            for n,p in named.items():
                assert torch.isfinite(p).all()
                branch='fm' if '_moe_gen' in n else 'ar' if 'lora_' in n else 'other'
                changed[branch]+=int(not torch.equal(p.detach().cpu(),initial[n]))
            initial.clear()
        record=dict(step=step,next_draw=next_draw,seconds=seconds,prepare_seconds=prep_seconds,schedule_seconds=schedule_seconds,lr=opt.param_groups[0]['lr'],grad_norm=norm.item(),gradient_tensors=len(params),changed=changed,peak_GiB=torch.cuda.max_memory_allocated()/2**30,reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        with (work/f'timing-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if rank==6:
            losses=[rows.pop(k) for k in range(draw_start,next_draw)]
            with (work/'loss.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=sum(x[0] for x in losses)/size,ce=sum(x[1] for x in losses)/size,fm=sum(x[2] for x in losses)/size))+'\n')
            print('optimizer_step',step,seconds,flush=True)
        if step%25==0 or step in (354,end_step):
            save_start=time.monotonic()
            path=work/'checkpoints'/f'step-{step:06d}';path.mkdir(exist_ok=True)
            weights={n:p.detach().cpu() for n,p in named.items()}
            optimizer={n:{k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in opt.state[p].items()} for n,p in named.items()}
            ck=path/f'rank{rank}.pt'
            torch.save(dict(step=step,next_draw=next_draw,params=weights,optimizer=optimizer,groups={n:{k:opt.param_groups[0][k] for k in ('lr','betas','eps','weight_decay')} for n in named},scheduler=dict(source_state=source_scheduler,updates_since350=step-350,equivalent_step=equivalent,lr=opt.param_groups[0]['lr']),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device)),ck.with_suffix('.tmp'))
            ck.with_suffix('.tmp').replace(ck)
            adapter={n:v.contiguous() for n,v in weights.items() if 'lora_' in n and '_moe_gen' not in n}
            shards=[None]*7 if rank==0 else None;dist.gather_object(adapter,shards,dst=0,group=control)
            if rank==0:
                from safetensors.torch import save_file
                merged={k:v for shard in shards for k,v in shard.items()}
                assert len(merged)==288 and all(torch.isfinite(v).all() for v in merged.values())
                save_file(merged,str(path/'ar-adapter.safetensors'))
                assert all((path/f'rank{r}.pt').stat().st_size for r in range(7))
                assert (path/'producer-rng.pt').exists()
                (path/'ready.json').write_text(json.dumps(dict(step=step,next_draw=next_draw,adapter_tensors=288,world_size=7)))
            dist.barrier();torch.cuda.synchronize()
            with (work/f'saves-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,seconds=time.monotonic()-save_start))+'\n')
    dist.destroy_process_group()
if __name__=='__main__':run('native')
