"""Model-independent data boundary: future images never enter policy inputs."""
def select(record, draw):
    assert record['id']==draw['id']
    view=record['views'][draw['level']]
    policy={'current_image':record['current_image'], 'instruction':view['instruction'],
            'history':record['history'], 'level':draw['level']}
    target={'assistant':view[draw['variant']], 'next_image':record['next_image'],
            'fm_mask':record['fm_mask']}
    assert target['assistant']
    assert target['fm_mask']==int(target['next_image'] is not None)
    return policy,target

def mixed_loss(ce_per_sample, fm_per_sample, mask, fm_weight=1.0):
    """Same-sized DDP batches: local mean + DDP average = global sample mean.

    Do not normalize by number of eligible transitions: it would amplify sparse
    transitions. The backend must supply finite per-sample FM losses and keep
    the same graph/collective structure on all ranks, including all-zero masks.
    """
    import torch
    assert ce_per_sample.ndim==fm_per_sample.ndim==mask.ndim==1
    assert ce_per_sample.shape==fm_per_sample.shape==mask.shape
    assert torch.isfinite(ce_per_sample).all() and torch.isfinite(fm_per_sample).all()
    assert ((mask==0)|(mask==1)).all()
    return ce_per_sample.mean()+fm_weight*(fm_per_sample*mask).mean()
