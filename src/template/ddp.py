import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


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


class DdpManager:
    def __init__(self, config):
        self.enabled = int(os.environ.get("RANK", -1)) != -1
        if self.enabled:
            self.config = config
            init_process_group(backend=config.backend)
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            assert (
                config.gradient_accumulation_steps % self.ddp_world_size == 0
            )
            self.gradient_accumulation_steps = (
                config.gradient_accumulation_steps // self.ddp_world_size
            )
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            self.device = config.device_type

    def wrap_model(self, model):
        if self.enabled:
            return DDP(model, device_ids=[self.ddp_local_rank])
        return model

    @staticmethod
    def destroy():
        destroy_process_group()
