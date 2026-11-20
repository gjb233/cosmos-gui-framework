"""GUI FM reductions, independent of heavyweight Cosmos training imports."""

import torch
import torch.distributed as dist

from .action_codec import ACTION_DIM, applicability


def global_sample_mean(values, active=None, *, group=None):
    """Value and gradient of a global mean under DDP/FSDP gradient averaging.

    Ranks with zero applicable samples still enter the same collective and
    retain a zero gradient connection. Collectives carry detached statistics.
    """
    if active is None:
        active = torch.ones_like(values, dtype=torch.bool)
    numerator = (values * active).sum()
    stats = torch.stack([numerator.detach(), active.sum().to(values)]).double()
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if world > 1:
        dist.all_reduce(stats, group=group)
    denominator = stats[1].clamp_min(1).to(values)
    gradient_term = numerator * world / denominator
    logged_mean = (stats[0] / stats[1].clamp_min(1)).to(values)
    return gradient_term + (logged_mean - gradient_term.detach())


def action_flow_loss(predictions, targets, clean_actions, *, group=None):
    if not (len(predictions) == len(targets) == len(clean_actions)) or not predictions:
        raise ValueError("Expected matching nonempty action batches")
    per_group = {"type": [], "xy": [], "direction": []}
    active_group = {key: [] for key in per_group}
    for prediction, target, clean in zip(predictions, targets, clean_actions, strict=True):
        if prediction.shape != target.shape or prediction.shape != clean.shape:
            raise ValueError("Action FM shapes differ")
        if prediction.ndim != 2 or prediction.shape[0] < 1:
            raise ValueError("GUI actions must be [T,D] with T >= 1")
        if prediction.shape[-1] < ACTION_DIM:
            raise ValueError("Action channels are fewer than 14")
        mask = applicability(clean.detach())
        error = (prediction.float() - target.float()).square()
        for name, start, end in (("type", 0, 8), ("xy", 8, 10), ("direction", 10, 14)):
            valid = mask[:, start:end]
            count = valid.sum()
            per_group[name].append((error[:, start:end] * valid).sum() / count.clamp_min(1))
            active_group[name].append(count > 0)
    losses = {
        name: global_sample_mean(torch.stack(values), torch.stack(active_group[name]), group=group)
        for name, values in per_group.items()
    }
    return sum(losses.values()), losses


def future_flow_per_sample(predictions, targets):
    values = []
    for prediction, target in zip(predictions, targets, strict=True):
        if prediction.ndim == 5 and prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)
        if target.ndim == 5 and target.shape[0] == 1:
            target = target.squeeze(0)
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] < 2:
            raise ValueError("Expected [C,T,H,W] vision latents with T >= 2")
        # First latent is observed. It must never contribute to the target loss.
        values.append((prediction[:, 1:].float() - target[:, 1:].float()).square().mean())
    if not values:
        raise ValueError("No future latent targets")
    return torch.stack(values)


def future_flow_loss(predictions, targets, *, group=None):
    return global_sample_mean(future_flow_per_sample(predictions, targets), group=group)


def plan_flow_per_sample(predictions, targets):
    values = []
    for prediction, target in zip(predictions, targets, strict=True):
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError("Expected matching [T,D] action-plan tensors")
        values.append((prediction.float() - target.float()).square().mean())
    if not values:
        raise ValueError("No action-plan targets")
    return torch.stack(values)


def recovered_x0_per_sample(xt, velocity, sigma):
    values = []
    for noisy, prediction, level in zip(xt, velocity, sigma, strict=True):
        if noisy.ndim == 5 and noisy.shape[0] == 1:
            noisy = noisy.squeeze(0)
        if prediction.ndim == 5 and prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)
        if prediction.ndim == noisy.ndim + 1 and prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)
        if noisy.ndim == 4 and level.ndim == 3 and level.shape[0] == noisy.shape[1]:
            level = level.unsqueeze(0)
        else:
            while level.ndim < noisy.ndim:
                level = level.unsqueeze(-1)
        values.append(noisy.float() - level.float() * prediction.float())
    return values


def plan_x0_per_sample(xt, velocity, sigma, targets):
    recovered = recovered_x0_per_sample(xt, velocity, sigma)
    return torch.stack(
        [
            torch.nn.functional.huber_loss(prediction, target.float())
            for prediction, target in zip(recovered, targets, strict=True)
        ]
    )


def inactive_action_flow_loss(predictions, targets, clean_actions, *, group=None):
    """Denoise unused semantic slots to zero without revealing applicability as input.

    The target is epsilon-minus-clean, not zero velocity. Otherwise these slots
    follow an unconstrained trajectory and feed back into the active action slots.
    Padded channels beyond the 14 semantic dimensions remain excluded.
    """
    if not predictions or not (len(predictions) == len(targets) == len(clean_actions)):
        raise ValueError("Expected matching nonempty action batches")
    values = []
    active = []
    for prediction, target, clean in zip(predictions, targets, clean_actions, strict=True):
        if prediction.shape != target.shape or prediction.shape != clean.shape or clean.shape[-1] < ACTION_DIM:
            raise ValueError("Action FM shapes differ")
        unused = ~applicability(clean.detach())[..., :ACTION_DIM]
        error = (prediction[..., :ACTION_DIM].float() - target[..., :ACTION_DIM].float()).square()
        count = unused.sum()
        values.append((error * unused).sum() / count.clamp_min(1))
        active.append(count > 0)
    return global_sample_mean(torch.stack(values), torch.stack(active), group=group)
