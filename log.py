from torch.utils.tensorboard import SummaryWriter
from data_logger import DataLogger

class TensorboardLogger(DataLogger):

    def __init__(self, writer=None):
        if writer is None:
            writer = SummaryWriter(log_dir="./logs")
        if not isinstance(writer, SummaryWriter):
            raise ValueError(
                "Only `SummaryWriter` class is allowed for the Tensorboard logger")
        self.writer = writer

    def log_value(self, name:str, value, step:int) -> None:
        if name == 'loss':
            print(name)
        self.writer.add_scalar(name, value, step)

    # writerを閉じる
    def close(self):
        self.writer.close()
