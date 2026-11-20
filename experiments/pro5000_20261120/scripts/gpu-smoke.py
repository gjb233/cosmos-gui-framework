import json
import torch
import torch.nn.functional as F
results=[]
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        x=torch.randn(64,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        loss=(x@x).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        results.append({'index':i,'name':torch.cuda.get_device_name(i),'capability':torch.cuda.get_device_capability(i),'finite_gradient':bool(torch.isfinite(x.grad).all())})
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':results},indent=2))
assert len(results)==8 and all(r['finite_gradient'] for r in results)
