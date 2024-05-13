RJUST_WIDTH = 20


def line_print(x):
    print(x, end=" ")


class StateLog:
    def __init__(self):
        self.previous_keys = None

    def log(self, dict):
        for k, v in dict.items():
            if "loss" in k:
                line_print(f"{k}: {v:.3f}")
            elif k == "optimization_step":
                line_print(f"step: {v}")
            else:
                line_print(f"{k}: {v}")
        line_print("\n")


state_log = StateLog()


def pretty_tokens(batch):
    print([[x.rjust(15) for x in sequence] for sequence in batch])
