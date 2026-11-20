import json,math,torch,os,re
from pathlib import Path
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from cosmos_framework.trainer import ImaginaireTrainer as Trainer
from cosmos_framework.gui_mot.native_model import GuiJointModel
from cosmos_framework.gui_mot.resumable_loader import GuiPackingDataLoader
ROOT=Path(__file__).resolve().parent
CK=ROOT.parent/'mai-ui-70284-input16384-1f1b-micro4-gb64-a21-gpu17-save50/1f1b/checkpoints/step-000150'
N=210962-9600
END=150+math.ceil(N/56)
def project(value,param):
    assert value.shape==param.shape
    if not isinstance(param,DTensor):return value.to(param.device)
    shape,offset=compute_local_shape_and_global_offset(param.shape,param.device_mesh,param.placements)
    part=value[tuple(slice(i,i+n) for i,n in zip(offset,shape))]
    assert part.shape==param.to_local().shape
    return DTensor.from_local(part.contiguous().to(param.device),param.device_mesh,param.placements,shape=param.shape,stride=param.stride())
def restore(model,container,sched):
    saved=torch.load(ROOT.parent/'migration-350.pt',map_location='cpu',weights_only=True)
    opt,=container.optimizers
    named={n.replace('_checkpoint_wrapped_module.',''):v for n,v in model.net.named_parameters() if v.requires_grad}
    assert set(named)==set(saved['params'])
    active={id(v):g for g in opt.param_groups for v in g['params']}
    for key,v in named.items():
        g=active[id(v)];state=saved['optimizer'][key]
        with torch.no_grad():v.copy_(project(saved['params'][key],v))
        opt.state[v]={k:project(state[k],v) for k in ('exp_avg','exp_avg_sq')}
        opt.state[v]['step']=state['step'].to(v.device if g.get('fused') or g.get('capturable') else 'cpu')
        for k,value in saved['groups'][key].items():g[k]=value
    sched.load_state_dict(saved['scheduler'])
    print('RESTORED_350',len(named),'next_draw',20800,flush=True)
origtrain=Trainer.train
def train(self,*args,**kwargs):
    base=self.checkpointer.load
    def load(model,opt,sched,scaler):
        iteration=base(model,opt,sched,scaler)
        assert iteration==0
        restore(model,opt,sched)
        return 350
    self.checkpointer.load=load
    return origtrain(self,*args,**kwargs)
Trainer.train=train
start=GuiPackingDataLoader.set_start_iteration
def offset(self,iteration):
    assert iteration>=1200
    start(self,(iteration-1200)*1)
GuiPackingDataLoader.set_start_iteration=offset
fetch=Trainer._fetch_data_batch
def fetch4(self,model,it):
    samples=[]
    for j in range(1):
        sample,stop=fetch(self,model,it)
        assert not stop
        samples.append(sample)
    batch=dict(samples[0]);batch['_micro4_samples']=samples
    return batch,False
Trainer._fetch_data_batch=fetch4
original=GuiJointModel.training_step
def forward4(self,batch,iteration):
    losses=[];back=[]
    samples=batch.pop('_micro4_samples')
    phase=getattr(self,"_micro4_phase",0)
    self._micro4_phase=(phase+1)%8
    for j,b in enumerate(samples):
        cap=b['ai_caption']
        while isinstance(cap,list) and len(cap)==1:cap=cap[0]
        slot=json.loads(cap)['draw_slot']
        expected=(iteration-150)*56+phase*7+torch.distributed.get_rank()+7*j
        assert slot==expected,(slot,expected)
        out,loss=original(self,b,iteration)
        losses.append(loss);back.append(out.get('_backward_loss',loss))
    out['_backward_loss']=torch.stack(back).mean()
    return out,torch.stack(losses).mean()
GuiJointModel.training_step=forward4
step=Trainer._optimizer_step
def step_lr(self,model,opt,sched,scaler,iteration):
    equivalent=150+min((iteration+1-150)*56,N)/64
    for g in opt.optimizers[0].param_groups:g['lr']=1e-5*(3298-equivalent)/(3298-33)
    return step(self,model,opt,sched,scaler,iteration)
Trainer._optimizer_step=step_lr

import time,pickle,random,numpy as np
import torch.distributed as dist
from cosmos_framework.gui_mot.training_artifacts import NativeTrainingArtifacts as A
base_fetch=Trainer._fetch_data_batch
def timed_fetch(self,model,it):
    if not getattr(self,'_rng_restored350',False):
        data=torch.load(ROOT.parent/'migration-350.pt',map_location='cpu',weights_only=True)['trainer']
        tag='rng_state_'+str(dist.get_rank())+'.'
        torch.set_rng_state(data[tag+'torch'])
        torch.cuda.set_rng_state(data[tag+'torch_cuda'])
        for key,fn in [('numpy',np.random.set_state),('random',random.setstate)]:
            n=int(data[tag+key+'_packed_len'])
            fn(pickle.loads(bytes(data[tag+key+'_packed_bytes'][:n].tolist())))
        self._rng_restored350=True
        model._resume_draw_checked=True
    if getattr(self,'_bench_started',None) is None:
        dist.barrier();torch.cuda.synchronize()
        self._bench_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    global bench_trainer
    bench_trainer=self
    return base_fetch(self,model,it)
Trainer._fetch_data_batch=timed_fetch
end_step=A.on_training_step_end
def timed_end(self,model,data_batch,output_batch,loss,iteration=0):
    end_step(self,model,data_batch,output_batch,loss,iteration=iteration)
    dist.barrier();torch.cuda.synchronize()
    seconds=time.monotonic()-bench_trainer._bench_started
    bench_trainer._bench_started=None
    root=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT'])
    with (root/f'timing-rank{dist.get_rank()}.jsonl').open('a') as f:
        f.write(json.dumps(dict(step=iteration,seconds=seconds,next_draw=9600+(iteration-150)*56,loss=float(loss),allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30))+'\n')
A.on_training_step_end=timed_end
base_forward=GuiJointModel.training_step
def audited_forward(self,batch,iteration):
    root=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT'])
    with (root/f'draws-rank{dist.get_rank()}.jsonl').open('a') as f:
        for b in batch['_micro4_samples']:
            c=b['ai_caption']
            while isinstance(c,list):c=c[0]
            c=json.loads(c)
            f.write(json.dumps(dict(draw=9600+c['draw_slot'],id=c['record_id'],level=c['level'],iteration=iteration))+'\n')
    return base_forward(self,batch,iteration)
GuiJointModel.training_step=audited_forward
