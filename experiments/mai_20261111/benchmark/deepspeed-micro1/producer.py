import sys,os,json
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
sys.path.insert(0,str(P))
import training_backend as b
import torch,torch.distributed as dist
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
if 'LOCAL_RANK' in os.environ:
 rank=int(os.environ['LOCAL_RANK'])
 torch.cuda.set_device(rank)
 free,total=torch.cuda.mem_get_info(rank)
 budget=total
 assert budget>20*1024**3,('insufficient budget',rank,budget)
 torch.cuda.set_per_process_memory_fraction(1.0,rank)
 print('MAI_ORIGINAL_RES_GPU_BUDGET',rank,budget,flush=True)
from cosmos_framework.gui_mot.training_artifacts import NativeTrainingArtifacts as A
def local(p):return p.to_local() if hasattr(p,'to_local') else p
def record(name,obj):
 root=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT']);root.mkdir(parents=True,exist_ok=True)
 with (root/(name+'-rank'+str(dist.get_rank())+'.jsonl')).open('a') as f:f.write(json.dumps(obj)+'\n')
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import misc
import time
original_train=ImaginaireTrainer.train
def train(self,model,dataloader_train,*args,**kwargs):
    self._benchmark_loader=dataloader_train
    global trainer
    trainer=self
    return original_train(self,model,dataloader_train,*args,**kwargs)
ImaginaireTrainer.train=train
class Captured(Exception):pass
def prepare_inputs(callback,model,iteration=0):
    import json,random,pickle,numpy as np
    assert iteration==0
    dest=P/'stream';dest.mkdir(exist_ok=True)
    migration=torch.load(P/'migration-350.pt',map_location='cpu',weights_only=True)
    rngs={r:{k.removeprefix(f'rng_state_{r}.'):v for k,v in migration['trainer'].items() if k.startswith(f'rng_state_{r}.')} for r in range(7)}
    del migration
    def restore_rng(state):
        torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['torch_cuda'])
        for key,fn in [('numpy',np.random.set_state),('random',random.setstate)]:
            fn(pickle.loads(bytes(state[key+'_packed_bytes'][:int(state[key+'_packed_len'])].tolist())))
    def capture_rng():
        state={'torch':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state()}
        for key,obj in [('numpy',np.random.get_state()),('random',random.getstate())]:
            raw=pickle.dumps(obj)
            state[key+'_packed_len']=torch.tensor(len(raw))
            state[key+'_packed_bytes']=torch.tensor(list(raw),dtype=torch.uint8)
        return state
    def network(*args,**kwargs):
        state['raw']=dict(args=args,kwargs=kwargs)
        return {}
    def objective(out_net,data_batch_packed,gen_data_noised,timesteps,is_image_batch,**kwargs):
        torch.save(dict(raw=state['raw'],target=dict(arguments=dict(data_batch_packed=data_batch_packed,gen_data_noised=gen_data_noised,timesteps=timesteps,is_image_batch=is_image_batch,kwargs=kwargs),fm_mask=list(model._mixed_fm_mask))),dest/f'{k}.tmp')
        (dest/f'{k}.tmp').rename(dest/f'{k}.pt')
        raise Captured()
    model.net.forward=network;model._compute_losses=objective;model.train()
    trainer._benchmark_loader.set_start_iteration(20800)
    it=iter(trainer._benchmark_loader)
    expected=[json.loads(line) for phase in (1,2) for line in (P/'artifacts'/f'schedule-phase{phase}.jsonl').open()]
    assert len(expected)==210962
    (dest/'ready').touch()
    for k in range(20800,21024):
        wave=(k-20800)//7
        while not (dest/f'request-{wave}').exists():
            if (dest/'abort').exists():raise RuntimeError('pipeline stopped')
            time.sleep(.05)
        t0=time.monotonic();state={}
        batch,stop=trainer._fetch_data_batch(model,it);assert not stop
        batch=misc.to(batch,device='cuda');trainer._cp_data_window.store_device_batch(batch)
        cap=batch['ai_caption']
        while isinstance(cap,list):cap=cap[0]
        cap=json.loads(cap)
        assert (cap['record_id'],cap['level'])==(expected[k]['id'],expected[k]['level']),(k,cap,expected[k])
        logical_rank=(k-9600)%7
        restore_rng(rngs[logical_rank])
        try:model.training_step(batch,350+(k-20800)//56)
        except Captured:pass
        else:raise RuntimeError('native packing did not execute')
        rngs[logical_rank]=capture_rng()
        with (dest/'draws.jsonl').open('a') as f:f.write(json.dumps(dict(draw=k,id=cap['record_id'],level=cap['level'],logical_rng_rank=logical_rank,seconds=time.monotonic()-t0))+'\n')
        if (k+1-20800)%56==0 or k+1==len(expected):
            step=350+(k+1-20800+55)//56
            if step%25==0 or step in (354,3746):
                path=P/'checkpoints'/f'step-{step:06d}';path.mkdir(exist_ok=True)
                torch.save(dict(rng_states=rngs,next_draw=k+1,step=step),path/'producer-rng.tmp')
                (path/'producer-rng.tmp').replace(path/'producer-rng.pt')
    raise SystemExit(0)
A.on_train_start=prepare_inputs
b.main()
