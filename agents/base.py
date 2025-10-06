from abc import ABC, abstractmethod
from typing import Any

class Agent(ABC):
    @abstractmethod
    def run(self,data,*arg,**kwargs)-> Any:
        """Takes input data and returns processed output"""
