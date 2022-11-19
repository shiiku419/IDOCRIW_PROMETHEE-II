import abc


class DataLogger(abc.ABC):
    def __dell__(self):
        self.close()

    @abc.abstractmethod
    def close(self) -> None:
        ...

    @abc.abstractmethod
    def log_value(self, name, value, step) -> None:
        ...
