import os
import torch
from types import SimpleNamespace
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Any


# def ddp_setup():
#     ddp = int(os.environ.get("RANK", -1)) != -1
#     if ddp:
#         init_process_group(backend=backend)
#         ddp_rank = int(os.environ["RANK"])
#         ddp_local_rank = int(os.environ["LOCAL_RANK"])
#         ddp_world_size = int(os.environ["WORLD_SIZE"])
#         device = f"cuda:{ddp_local_rank}"
#         torch.cuda.set_device(device)
#         master_process = ddp_rank == 0
#         seed_offset = ddp_rank
#         gradient_accumulation_steps //= ddp_world_size
#     else:
#         master_process = True
#         seed_offset = 0
#         ddp_world_size = 1

# ddp_enabled = int(os.environ.get("RANK", -1)) != -1

# default is parallel, single gpu is a special case of parallel


# class DdpManager:
#     def __init__(self, config):
#         self.enabled = int(os.environ.get("RANK", -1)) != -1
#         if self.enabled:
#             self.config = config
#             init_process_group(backend=config.backend)
#             self.ddp_rank = int(os.environ["RANK"])
#             self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
#             self.ddp_world_size = int(os.environ["WORLD_SIZE"])
#             self.device = f"cuda:{self.ddp_local_rank}"
#             torch.cuda.set_device(self.device)
#             self.master_process = self.ddp_rank == 0
#             self.seed_offset = self.ddp_rank
#             assert (
#                 config.gradient_accumulation_steps % self.ddp_world_size == 0
#             )
#             self.gradient_accumulation_steps = (
#                 config.gradient_accumulation_steps // self.ddp_world_size
#             )
#         else:
#             self.master_process = True
#             self.seed_offset = 0
#             self.ddp_world_size = 1
#             self.device = config.device_type
#
#     def wrap_model(self, model):
#         if self.enabled:
#             return DDP(model, device_ids=[self.ddp_local_rank])
#         res = SimpleNamespace()
#         res.module = model
#         return res
#
#     @staticmethod
#     def destroy():
#         destroy_process_group()


# https://github.com/Lightning-AI/pytorch-lightning/blob/df5dee674243e124a2bf34d9975dd586ff008d4b/src/lightning/pytorch/strategies/strategy.py#L628
class _ForwardRedirection:
    """Implements the `forward-redirection`.

    A method call to a wrapped module gets rerouted through the wrapper's `forward` method instead.

    """

    def __call__(
        self,
        wrapper_module,
        original_module,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ):
        """Reroutes a method call through the `wrapper_module`'s `forward` method.

        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method_name: The name of the method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.

        """
        assert method_name != "forward"
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            method = getattr(original_module, method_name)
            out = method(*_args, **_kwargs)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]

        wrapper_output = wrapper_module(*args, **kwargs)
        return wrapper_output
