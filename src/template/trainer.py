import torch
from types import SimpleNamespace

from pathlib import Path
import sys
from contextlib import nullcontext
from template.log import state_log


torch_type = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class Trainer:
    def __init__(self, config, data, model, ddp, checkpoint=None):
        self.config = config
        self.device_type = self.config.device_type
        self.ddp = ddp
        self.init_state(checkpoint)
        self.data = data
        model = model.to(self.ddp.device)
        self.model = self.ddp.wrap_model(model)
        self.init_optimizer()
        # autocast very slow with cpu and float16 therefore nullcontext
        self.ctx = (
            nullcontext()
            if self.config.device_type == "cpu"
            else torch.amp.autocast(
                device_type=self.device_type, dtype=torch_type[config.dtype]
            )
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(config.dtype == "float16")
        )

        del checkpoint

    def generate(self):
        for _ in range(5):
            x = self.data.encode("\n")
            x = torch.tensor(x, dtype=torch.long, device=self.config.device)[
                None, ...
            ]

            y = self.model.generate(x)
            print(y.shape)

            print(self.data.decode(y[0].tolist()))
            print("--------------")

    @staticmethod
    def iterable_to_device(iterable, device):
        return tuple(x.to(device) for x in iterable)

    def run(self):
        self.evaluation_step()
        self.evaluation_step_log()
        # todo: delete
        # self.sample()
        loader = self.data.train_loader()
        for epoch in range(self.config.epoch):
            self.state.epoch = epoch
            for step, data in enumerate(loader, start=1):
                # looks ugly but the logic is very easy to read
                self.state.step = step
                data = self.iterable_to_device(data, self.ddp.device)
                if self.ddp.enabled:
                    self.model.require_backward_grad_sync = (
                        self.state.step
                        % self.config.gradient_accumulation_steps
                        == 0
                    )
                self.forward_backward_step(data)
                if self.should_optimize():
                    self.optimize()
                    if self.should_evaluate():
                        self.evaluate()
                        if self.should_save_checkpoint():
                            self.save_checkpoint()

            self.optimize()
            self.evaluate()
            if self.should_save_checkpoint():
                self.save_checkpoint()

    # todo: delete
    def load_checkpoint(self):
        experiment_path = Path(sys.argv[2])
        checkpoint_path = experiment_path / "checkpoint.ckpt"
        # why cpu
        # https://github.com/pytorch/pytorch/issues/7415#issuecomment-693424574
        if checkpoint_path.exists():
            self.checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # todo: delete?
    def del_checkpoint(self):
        if hasattr(self, "checkpoint"):
            del self.checkpoint

    def init_state(self, checkpoint=None):
        if checkpoint is not None:
            self.state = checkpoint["state"]
            return
        self.state = SimpleNamespace()
        self.state.step = 0
        self.state.epoch = 0
        self.state.optimization_step = 0
        self.state.best_save_metric = -float("inf")

    def init_optimizer(self, checkpoint=None):
        self.optimizer = self.model.create_optimizer()
        if checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])

    def forward_backward_step(self, data):
        # pytorch super slow with autocast on cpu and bfloat16
        with self.ctx:
            _, loss = self.model.training_step(data)
        self.state.train_loss = round(loss.item(), 3)
        self.state.loss = loss / self.config.gradient_accumulation_steps
        self.scaler.scale(self.state.loss).backward()

    def handle_save_metric(self):
        self.state.save_metric = -self.state.eval_loss

    def should_optimize(self):
        return (self.state.step) % self.config.gradient_accumulation_steps == 0

    def optimize_step(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.optimization_step += 1

    def optimize_step_log(self):
        keys = "step", "epoch", "optimization_step", "train_loss"
        res = {}
        for k in keys[1:]:
            if not hasattr(self.state, k):
                assert False, f"{k} not in self.state"
            res[k] = getattr(self.state, k)
        # state_log.log(keys, vals)
        state_log.log(res)

    def optimize(self):
        self.optimize_step()
        self.optimize_step_log()

    def should_evaluate(self):
        return (
            self.state.optimization_step
        ) % self.config.eval_interval == 0 and self.ddp.master_process

    @torch.no_grad()
    def evaluation_step(self):
        self.model.eval()
        loader = self.data.validation_loader()
        losses = torch.zeros(self.config.eval_iters)
        self.state.eval_predictions = []
        self.state.eval_labels = []
        for i, data in enumerate(loader):
            data = self.iterable_to_device(data, self.ddp.device)
            with self.ctx:
                prediction, eval_loss = self.model.evaluation_step(data)
            losses[i] = eval_loss
            # may need input, so append all data
            self.state.eval_labels.append(data)
            self.state.eval_predictions.append(prediction)
            if i == self.config.eval_iters - 1:
                break
        self.state.metric = self.data.compute_metric(
            self.state.eval_predictions, self.state.eval_labels
        )
        self.state.eval_loss = round(losses.mean().item(), 4)
        self.handle_save_metric()
        self.model.train()

    def evaluation_step_log(self):
        keys = [
            "mode",
            "epoch",
            "optimization_step",
            "eval_loss",
            "metric",
        ]
        res = {"step": "evaluation"}
        for k in keys[1:]:
            if not hasattr(self.state, k):
                assert False, f"{k} not in self.state"
            res[k] = getattr(self.state, k)
        state_log.log(res)

    def evaluate(self):
        self.evaluation_step()
        self.evaluation_step_log()

    def should_save_checkpoint(self):
        if self.state.best_save_metric >= self.state.save_metric:
            return False
        self.state.best_save_metric = self.state.save_metric
        print(
            f"new best metric: {self.state.best_save_metric}, save checkpoint..."
        )
        return True

    def save_checkpoint(self):
        checkpoint = {
            "model": (
                self.model.module.state_dict()
                if self.ddp.enabled
                else self.model.state_dict()
            ),
            "optimizer": self.optimizer.state_dict(),
            "state": self.state,
        }

        checkpoint_path = Path(sys.argv[2] + "/checkpoint.ckpt")
        print(f"saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
