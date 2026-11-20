import json, torch
import torch.distributed as dist
import os
if 'LOCAL_RANK' in os.environ:
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    x=torch.tensor([rank+1.0],device='cuda')
    dist.all_reduce(x)
    assert x.item()==36, x
    if rank==0: print('NCCL_8_GPU_ALL_REDUCE_OK',flush=True)
    dist.destroy_process_group()
else:
    result={}
    for name in ['flash_attn','natten','transformer_engine.pytorch','peft']:
        try:
            __import__(name)
            result[name]='import OK'
        except Exception as e: result[name]=repr(e)
    try:
        from flash_attn import flash_attn_func
        q=torch.randn(1,128,4,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        flash_attn_func(q,q,q).float().sum().backward()
        torch.cuda.synchronize()
        result['flash_attn_forward_backward']=bool(torch.isfinite(q.grad).all())
    except Exception as e: result['flash_attn_forward_backward']=repr(e)
    print(json.dumps(result,indent=2))
