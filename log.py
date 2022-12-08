from torch.utils.tensorboard import SummaryWriter
from data_logger import DataLogger


class TensorboardLogger(DataLogger):

    def __init__(self, writer=None):
        if writer is None:
            writer = SummaryWriter(log_dir="./logs/exp2")
        if not isinstance(writer, SummaryWriter):
            raise ValueError(
                "Only `SummaryWriter` class is allowed for the Tensorboard logger")
        self.writer = writer

    def log_value(self, name: str, value, step: int) -> None:
        self.writer.add_scalars(name, value, step)

    def close(self):
        self.writer.close()
