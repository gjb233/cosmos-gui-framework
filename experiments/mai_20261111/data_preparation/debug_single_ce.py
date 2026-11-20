# Single-card CE smoke only; does not validate the eight-source training backend.
import argparse,json,os,subprocess,fcntl,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu-uuid',required=True);args=parser.parse_args()
    lock=(R/'single-card-debug.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def free():
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        rows=[x.split(', ') for x in gpu.splitlines()]
        row=next(x for x in rows if x[0]==args.gpu_uuid)
        assert 'RTX A6000' in row[1] and int(row[2])<=100 and int(row[3])==0
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        assert args.gpu_uuid not in procs
    free();time.sleep(1);free()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_uuid
    os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(0,'/rscratch/wzzheng/code/cosmos-gui-framework')
    sys.path.insert(0,'/rscratch/wzzheng/experiments/gui-libra-low-ce-only1000-micro2-a27-20261022-control')
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from libra_ce1000_fresh_lora import install_fresh,export
    sample=Path('/rscratch/wzzheng/experiments/gui-libra-low-terminal-cefm500-a22-20261014-v1/actual-training-example.json')
    row=json.loads(sample.read_text());assert row['only_assistant_target_and_EOS_have_loss']
    model_path='/rscratch/wzzheng/code/cosmos-gui/.local/models/GUI-Libra-8B'
    out=R/'debug'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    (out/'started.json').write_text(json.dumps({'pid':os.getpid(),'gpu_uuid':args.gpu_uuid,'sample':str(sample),'scope':'existing AC training fixture CE smoke; not new mixed-data validation'}))
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
    initial=install_fresh(model);model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
    messages=[{'role':'system','content':row['system']},{'role':'user','content':[{'type':'image','image':row['image']},{'type':'text','text':row['user_text']}]}]
    im=Image.open(row['image']).convert('RGB')
    prefix=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    full=processor.apply_chat_template(messages+[{'role':'assistant','content':row['assistant_target']}],tokenize=False,add_generation_prompt=False)
    a=processor(text=[prefix],images=[im],return_tensors='pt')
    b=processor(text=[full],images=[im],return_tensors='pt')
    n=a['input_ids'].shape[1];assert torch.equal(a['input_ids'],b['input_ids'][:,:n])
    labels=b['input_ids'].clone();labels[:,:n]=-100;assert (labels!=-100).any()
    batch={k:v.cuda() for k,v in b.items()};params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.99),weight_decay=0)
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,labels=labels.cuda(),use_cache=False).loss
    assert torch.isfinite(loss);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    norm=torch.nn.utils.clip_grad_norm_(params,1.0);assert torch.isfinite(norm);opt.step()
    after=export(model);assert all(torch.isfinite(v).all() for v in after.values())
    assert any(not torch.equal(initial[k],v) for k,v in after.items())
    (out/'complete.json').write_text(json.dumps({'passed':True,'steps':1,'loss':float(loss.detach()),'grad_norm':float(norm),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'gpu_uuid':args.gpu_uuid,'mixed_data_validated':False,'fm_forward_validated':False},indent=2))
if __name__=='__main__': main()
