"""Load a checkpoint including the parts PEFT does not save.

`save_pretrained` writes only adapter tensors -- its own docstring says "not
the parameters of the base model". v2 also trains `A_log`, `dt_bias`, the
convolutions and the norms directly, which LoRA cannot reach, and those are
written beside the adapter as `direct.pt`. Loading the adapter alone would
silently discard them and make the configuration look inert.

Several adapters can share one base model, but the direct tensors are base
weights, so only one set can be resident at a time. `DirectWeights` keeps the
originals and swaps them when the active adapter changes.
"""
from pathlib import Path

import torch


class DirectWeights:
    """Swap directly-trained base tensors in and out as adapters change."""

    def __init__(self, model):
        self.model = model
        self.original = {}

    def _params(self):
        return dict(self.model.named_parameters())

    def apply(self, ckpt_dir) -> int:
        """Load `<ckpt_dir>/direct.pt` if present; restore the base otherwise.

        Returns the number of tensors overwritten.
        """
        params = self._params()
        # Put back whatever the previous checkpoint changed.
        for k, v in self.original.items():
            if k in params:
                params[k].data.copy_(v.to(params[k].device, params[k].dtype))
        self.original.clear()
        if ckpt_dir is None:
            return 0
        path = Path(ckpt_dir) / "direct.pt"
        if not path.exists():
            return 0
        saved = torch.load(path, map_location="cpu")
        missing = [k for k in saved if k not in params]
        if missing:
            raise KeyError(
                f"{path} holds {len(missing)} tensors absent from the model, "
                f"e.g. {missing[:3]} -- the checkpoint and the model disagree"
            )
        for k, v in saved.items():
            self.original[k] = params[k].detach().cpu().clone()
            params[k].data.copy_(v.to(params[k].device, params[k].dtype))
        return len(saved)
