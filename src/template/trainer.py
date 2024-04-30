import torch
from types import SimpleNamespace
import json
import tomllib

from pathlib import Path
import sys

RJUST_WIDTH = 20


class StateLog:
    def __init__(self):
        self.previous_keys = None

    def log(self, keys, vals):
        vals = "".join([f"{v}".rjust(RJUST_WIDTH) for v in vals])
        if keys == self.previous_keys:
            print(vals)
            return
        self.previous_keys = keys
        keys = "".join([k.rjust(RJUST_WIDTH) for k in keys])
        print(f"\n{keys}\n{vals}")


state_log = StateLog()


def pretty_tokens(batch):
    print([[x.rjust(15) for x in sequence] for sequence in batch])


def load_config_dict(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


class Trainer:
    def __init__(self, config, data, model, checkpoint=None):
        self.config = config
        self.device = self.config.device
        self.init_state(checkpoint)
        self.data = data
        self.model = model.to(self.device)
        self.init_optimizer()
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

    def run(self):
        self.evaluation_step()
        self.evaluation_step_log()
        # self.sample()
        loader = self.data.train_loader()
        for epoch in range(self.config.epoch):
            self.state.epoch = epoch
            for step, data in enumerate(loader):
                # looks ugly but the logic is very easy to read
                self.state.step = step + 1
                # data = self.iterable_to_device(data, self.device)
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

    def load_checkpoint(self):
        experiment_path = Path(sys.argv[2])
        checkpoint_path = experiment_path / "checkpoint.ckpt"
        # why cpu
        # https://github.com/pytorch/pytorch/issues/7415#issuecomment-693424574
        if checkpoint_path.exists():
            self.checkpoint = torch.load(checkpoint_path, map_location="cpu")

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
        # with torch.autocast(
        #     device_type=self.config.device, dtype=torch.bfloat16
        # ):
        _, loss = self.model.training_step(data)
        self.state.train_loss = round(loss.item(), 3)
        self.state.loss = loss / self.config.gradient_accumulation_steps
        self.state.loss.backward()

    def handle_save_metric(self):
        self.state.save_metric = -self.state.eval_loss

    def should_optimize(self):
        return (self.state.step) % self.config.gradient_accumulation_steps == 0

    def optimize_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.optimization_step += 1

    def optimize_step_log(self):
        keys = "step", "epoch", "optimization_step", "train_loss"
        vals = ["optimization"]
        for k in keys[1:]:
            if not hasattr(self.state, k):
                assert False, f"{k} not in self.state"
            vals.append(getattr(self.state, k))
        state_log.log(keys, vals)

    def optimize(self):
        self.optimize_step()
        self.optimize_step_log()

    def should_evaluate(self):
        return (self.state.optimization_step) % self.config.eval_interval == 0

    @torch.no_grad()
    def evaluation_step(self):
        self.model.eval()
        loader = self.data.validation_loader()
        losses = torch.zeros(self.config.eval_iters)
        self.state.eval_predictions = []
        self.state.eval_labels = []
        for i, data in enumerate(loader):
            # with torch.autocast(
            #     device_type=self.config.device, dtype=torch.bfloat16
            # ):
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
        vals = ["evaluation"]
        for k in keys[1:]:
            if not hasattr(self.state, k):
                assert False, f"{k} not in self.state"
            vals.append(getattr(self.state, k))
        state_log.log(keys, vals)

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
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state": self.state,
        }

        checkpoint_path = Path(sys.argv[2] + "/checkpoint.ckpt")
        print(f"saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
