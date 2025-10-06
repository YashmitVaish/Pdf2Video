from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def run(self, data):
        """Takes input data and returns processed output"""
        pass
