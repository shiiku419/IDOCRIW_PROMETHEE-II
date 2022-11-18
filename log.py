from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger():

    def __init__(self, writer=None):
        if writer is None:
            writer = SummaryWriter(log_dir="./logs")
        if not isinstance(writer, SummaryWriter):
            raise ValueError(
                "Only `SummaryWriter` class is allowed for the Tensorboard logger")
        self.writer = writer

    def log_value(self, name, value, step) -> None:
        self.writer.add_scalar(name, value, step)

    # writerを閉じる
    def close(self):
        self.writer.close()
