"""Tensor-only boundary codec. Metadata stays separate for each microbatch."""
import dataclasses
import torch

class BoundaryCodec:
    def __init__(self, schema):
        self.schema = schema

    @classmethod
    def flatten(cls, value):
        tensors = []
        def walk(x):
            if isinstance(x, torch.Tensor):
                index = len(tensors)
                tensors.append(x)
                return ("tensor", index)
            if isinstance(x, torch.device):
                return ("device", str(x))
            if type(x).__module__ == 'cosmos_framework.model.generator.mot.attention' and type(x).__name__ == 'SplitInfo':
                return ('split_info', type(x), tuple((k, walk(v)) for k,v in vars(x).items()))
            if dataclasses.is_dataclass(x) and not isinstance(x, type):
                return ("dataclass", type(x), tuple((f.name, walk(getattr(x, f.name))) for f in dataclasses.fields(x)))
            if isinstance(x, dict):
                if not all(isinstance(k, str) for k in x):
                    raise TypeError("Boundary mapping keys must be strings")
                return ("dict", tuple((k, walk(v)) for k,v in x.items()))
            if type(x) in (list, tuple):
                return ("list" if type(x) is list else "tuple", tuple(walk(v) for v in x))
            if x is None or type(x) in (bool,int,float,str):
                return ("constant", x)
            raise TypeError("Unsupported boundary object: "+type(x).__module__+"."+type(x).__qualname__)
        schema = walk(value)
        return cls(schema), tuple(tensors)

    def rebuild(self, tensors, device=None):
        used = []
        def walk(s):
            tag = s[0]
            if tag == "tensor":
                used.append(s[1])
                return tensors[s[1]]
            if tag == "constant": return s[1]
            if tag == "device": return torch.device(device if device is not None else s[1])
            if tag == "dict": return {k:walk(v) for k,v in s[1]}
            if tag == "list": return [walk(v) for v in s[1]]
            if tag == "tuple": return tuple(walk(v) for v in s[1])
            if tag == 'split_info':
                result = object.__new__(s[1])
                for k,v in s[2]: setattr(result,k,walk(v))
                return result
            if tag == "dataclass": return s[1](**{k:walk(v) for k,v in s[2]})
            raise ValueError(tag)
        result = walk(self.schema)
        if sorted(used) != list(range(len(tensors))):
            raise ValueError("Boundary tensor count mismatch")
        return result
