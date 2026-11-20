"""Rank-local placement after loading the unwrapped MAI model on CPU.
Unused CPU weights are initialization residue, never activation offload.
Caller must build its optimizer from owned_trainable_parameters, not net.parameters.
"""
import re
import torch
from joint_stage import JointDecoderStage
BOUNDS=(0,3,9,15,21,26,31,36)
FRONTEND=("time_embedder.","vae2llm.")
HEAD=("llm2vae.",)

def trainable_owner(name):
    match=re.match(r"language_model.model.layers.(\d+)\.",name)
    if match:
        layer=int(match.group(1))
        for rank in range(len(BOUNDS)-1):
            if BOUNDS[rank]<=layer<BOUNDS[rank+1]:return rank
        raise ValueError("Layer outside partition")
    if name.startswith(FRONTEND):return 0
    if name.startswith(HEAD):return len(BOUNDS)-2
    raise ValueError("Unclassified trainable parameter: "+name)

def place_rank(network, rank, device):
    if rank not in range(len(BOUNDS)-1):raise ValueError(rank)
    device=torch.device(device)
    if device.type!="cuda":raise ValueError("GPU stage required")
    params=list(network.named_parameters())
    if any(p.device.type!="cpu" for _,p in params):
        raise ValueError("Load original weights on CPU before rank placement")
    ownership={n:trainable_owner(n) for n,p in params if p.requires_grad}
    # Decoder storage is strictly local. Endpoint ranks conservatively retain
    # frozen non-decoder modules until exact endpoint pruning is validated.
    def local(name,trainable=False):
        if trainable:return ownership[name]==rank
        m=re.match(r"language_model.model.layers.(\d+)\.",name)
        if m:return BOUNDS[rank]<=int(m.group(1))<BOUNDS[rank+1]
        return rank in (0,len(BOUNDS)-2)
    for name,p in params:
        if local(name,p.requires_grad):
            p.data=p.data.to(device=device,dtype=torch.float32 if p.requires_grad else torch.bfloat16)
    for name,b in list(network.named_buffers()):
        if local(name):
            parent,_,key=name.rpartition(".")
            module=network.get_submodule(parent) if parent else network
            module._buffers[key]=b.to(device)
    decoder=JointDecoderStage(network.language_model.model.layers,BOUNDS[rank],BOUNDS[rank+1])
    owned={n:p for n,p in params if p.requires_grad and ownership[n]==rank}
    assert all(p.device==device for p in owned.values())
    assert all(p.device==device for p in decoder.parameters())
    return decoder,owned,ownership
