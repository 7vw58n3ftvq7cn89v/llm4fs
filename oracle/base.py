# src/backend_new/core/oracle/base.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseOracle(ABC):
    def __init__(self, name: str="BaseOracle"):
        self.name = name
        # self.task_type = "classification"

    @abstractmethod
    def train(self, data: pd.DataFrame, target_col: str) -> float:
        """训练模型并返回评估指标"""
        pass