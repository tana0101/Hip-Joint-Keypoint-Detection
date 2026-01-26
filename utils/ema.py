import copy
import torch

class ModelEMA:
    """
    Exponential Moving Average (EMA) of model parameters + buffers.
    Supports ramp-up EMA decay: start from a smaller decay then increase to target_decay.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_decay: float = 0.997,
        device: str | torch.device | None = None,
        rampup_steps: int = 1000,
        start_decay: float = 0.95,
    ):
        """
        Args:
            model: the training model
            target_decay: final EMA decay (e.g., 0.995~0.999 for small/noisy data)
            device: store EMA weights on this device (e.g., 'cpu' to save GPU memory)
            rampup_steps: number of updates to ramp start_decay -> target_decay
            start_decay: initial EMA decay during early training (smaller = follows faster)
        """
        assert 0.0 < start_decay < 1.0
        assert 0.0 < target_decay < 1.0
        assert rampup_steps >= 0

        self.target_decay = float(target_decay)
        self.start_decay = float(start_decay)
        self.rampup_steps = int(rampup_steps)
        self.updates = 0  # counts update() calls

        # Deep copy of model for EMA
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.device = device
        if self.device is not None:
            self.ema.to(self.device)

    def _get_decay(self) -> float:
        """Linearly ramp decay from start_decay to target_decay over rampup_steps."""
        if self.rampup_steps <= 0:
            return self.target_decay
        t = min(1.0, self.updates / float(self.rampup_steps))
        return self.start_decay + t * (self.target_decay - self.start_decay)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """
        Update EMA weights.
        Call this AFTER optimizer.step().
        """
        self.updates += 1
        d = self._get_decay()

        msd = model.state_dict()
        esd = self.ema.state_dict()

        for k, v in esd.items():
            src = msd[k]
            if v.dtype.is_floating_point:
                # EMA for float tensors (params + float buffers)
                v.mul_(d).add_(src, alpha=1.0 - d)
            else:
                # Copy for non-float buffers (e.g., num_batches_tracked)
                v.copy_(src)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict, strict=True)
