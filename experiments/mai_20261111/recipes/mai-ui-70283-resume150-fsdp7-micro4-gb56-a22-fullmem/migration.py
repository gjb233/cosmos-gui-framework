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
    opt,=container.optimizers
    named={n.replace('_checkpoint_wrapped_module.',''):v for n,v in model.net.named_parameters() if v.requires_grad}
    assert len(named)==584,(len(named),list(named)[:8])
    active={id(v):g for g in opt.param_groups for v in g['params']}
    restored=set()
    for rank in (0,1):
        saved=torch.load(CK/f'rank{rank}.pt',map_location='cpu',weights_only=True)
        group,=saved['optimizer']['param_groups']
        assert saved['step']==150 and saved['next_draw']==9600 and len(group['params'])==len(saved['params'])
        for (name,value),idx in zip(saved['params'].items(),group['params']):
            if name.startswith('endpoint.'):key=name.removeprefix('endpoint.')
            else:
                layer,tail=name.split('decoder.flat_stage.decoder.layers.')[1].split('.',1)
                key=f'language_model.model.layers.{int(layer)+17*rank}.{tail}'
            assert key not in restored and key in named,(key,list(named)[:5])
            v=named[key];g=active[id(v)];state=saved['optimizer']['state'][idx]
            with torch.no_grad():v.copy_(project(value,v))
            assert int(state['step'])==150
            opt.state[v]={k:project(state[k],v) for k in ('exp_avg','exp_avg_sq')}
            opt.state[v]['step']=state['step'].to(v.device if g.get('fused') or g.get('capturable') else 'cpu')
            for k in ('lr','betas','eps','weight_decay'):g[k]=group[k]
            restored.add(key)
    assert restored==set(named)
    print('RESTORED_150',len(restored),'next_draw',9600,flush=True)
origtrain=Trainer.train
def train(self,*args,**kwargs):
    base=self.checkpointer.load
    def load(model,opt,sched,scaler):
        iteration=base(model,opt,sched,scaler)
        assert iteration==0
        restore(model,opt,sched)
        return 150
    self.checkpointer.load=load
    return origtrain(self,*args,**kwargs)
Trainer.train=train
start=GuiPackingDataLoader.set_start_iteration
def offset(self,iteration):
    assert iteration>=300
    start(self,(iteration-300)*4)
GuiPackingDataLoader.set_start_iteration=offset
fetch=Trainer._fetch_data_batch
def fetch4(self,model,it):
    samples=[]
    for j in range(4):
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
    self._micro4_phase=1-phase
    for j,b in enumerate(samples):
        cap=b['ai_caption']
        while isinstance(cap,list) and len(cap)==1:cap=cap[0]
        slot=json.loads(cap)['draw_slot']
        expected=(iteration-150)*56+phase*28+torch.distributed.get_rank()+7*j
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
