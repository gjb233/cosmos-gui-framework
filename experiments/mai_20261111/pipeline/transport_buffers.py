"""Fixed-capacity transport, with trimming before model compute.
Only transport tensors are padded. No padded rows enter MAI attention/loss.
"""
import math
import torch
import torch.nn.functional as F

class TensorEnvelope:
    def __init__(self, shapes, dtypes, capacities=None):
        self.shapes = tuple(tuple(s) for s in shapes)
        self.dtypes = tuple(dtypes)
        if len(self.shapes) != len(self.dtypes):
            raise ValueError("Shape/dtype count mismatch")
        sizes = tuple(math.prod(s) for s in self.shapes)
        self.capacities = tuple(sizes if capacities is None else capacities)
        if len(sizes) != len(self.capacities) or any(n > c or c < 0 for n,c in zip(sizes,self.capacities)):
            raise ValueError("Invalid or insufficient transport capacity")

    def pack(self, tensors):
        if len(tensors) != len(self.shapes):
            raise ValueError("Tensor count mismatch")
        wires = []
        for tensor,shape,dtype,capacity in zip(tensors,self.shapes,self.dtypes,self.capacities):
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError("Tensor disagrees with this microbatch envelope")
            wires.append(F.pad(tensor.reshape(-1), (0,capacity-tensor.numel())))
        return tuple(wires)

    def unpack(self, wires):
        if len(wires) != len(self.shapes):
            raise ValueError("Wire count mismatch")
        result = []
        for wire,shape,dtype,capacity in zip(wires,self.shapes,self.dtypes,self.capacities):
            if wire.ndim != 1 or wire.numel() != capacity or wire.dtype != dtype:
                raise ValueError("Wire capacity/dtype mismatch")
            result.append(wire[:math.prod(shape)].reshape(shape))
        return tuple(result)
