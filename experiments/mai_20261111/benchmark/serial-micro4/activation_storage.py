import torch
import time
stats = dict(offloaded_bytes=0, offloaded_tensors=0, saved_bytes=0, started=time.monotonic())
def install(cls):
    original = cls.training_step
    def step(self, *args, **kwargs):
        free, total = torch.cuda.mem_get_info()
        # Keep most activations on GPU; leave workspace and all-gather headroom.
        threshold = max(16*2**30, min(32*2**30, torch.cuda.memory_allocated()+free-12*2**30))
        def pack(t):
            stats["saved_bytes"] += t.numel()*t.element_size()
            if type(t) is torch.Tensor and t.is_cuda and t.layout == torch.strided and not t.is_leaf and t.numel()*t.element_size() >= 2**20 and torch.cuda.memory_allocated() > threshold:
                cpu = t.detach().to(device='cpu', copy=True)
                stats['offloaded_bytes'] += t.numel()*t.element_size()
                stats['offloaded_tensors'] += 1
                return (t.device, cpu)
            return t
        def unpack(x):
            if isinstance(x, tuple):
                device, cpu = x
                return cpu.to(device=device)
            return x
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return original(self, *args, **kwargs)
    cls.training_step = step
