from abc import ABC, abstractmethod

class Step(ABC):
    @property
    @abstractmethod
    def command_name(self):
        pass

    @property
    @abstractmethod
    def timestamp(self):
        pass

    @abstractmethod
    def routine(self, _input):
        pass
