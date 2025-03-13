# src/backend_new/core/oracle/optimized_classifier.py
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import resample
from .base import BaseOracle

class OptimizedClassifierOracle(BaseOracle):
    """优化版分类器，包含数据平衡和集成学习"""
    
    def _preprocess_data(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
        """数据预处理
        
        Args:
            data (pd.DataFrame): 输入数据
            target_col (str): 目标列名
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: 处理后的数据集和特征列表
        """
        dataset = data.copy()
        columns = list(dataset.columns)
        columns.remove(target_col)
        dataset = dataset[dataset[target_col].notna()]
        
        # 数据预处理
        for col in columns:
            if dataset[col].dtype.name == 'object':
                # 对于类别型数据，先填充缺失值，再进行编码
                dataset[col] = dataset[col].fillna('MISSING')
                dataset[col] = dataset[col].astype('category')
                dataset[col] = dataset[col].cat.codes
            else:
                # 对于数值型数据，用中位数填充缺失值
                dataset[col] = dataset[col].fillna(dataset[col].median())
        

        # 最后检查确保没有任何 NaN 值
        assert not dataset.isnull().any().any(), "数据中仍然存在 NaN 值"
        return dataset, columns
    
    def _balance_dataset(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """平衡数据集
        
        Args:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 标签数据
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 平衡后的特征和标签数据
        """
        # 合并数据用于重采样
        train_data = pd.concat([X, y], axis=1)
        
        # 分离多数类和少数类
        df_majority = train_data[train_data[y.name] == 0]
        df_minority = train_data[train_data[y.name] == 1]
        
        # 下采样多数类
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=488,  # 与原代码保持一致
            random_state=123
        )
        
        # 合并平衡后的数据
        balanced_data = pd.concat([df_minority, df_majority_downsampled])
        
        # 返回特征和标签
        return (
            balanced_data.drop(columns=[y.name]),
            balanced_data[y.name]
        )
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算评估指标
        
        Args:
            y_true (np.ndarray): 真实标签
            y_pred (np.ndarray): 预测标签
            
        Returns:
            float: F1分数
        """
        return f1_score(y_true, y_pred, average='macro')
    
    def train(self, data: pd.DataFrame, target_col: str) -> float:
        """训练优化版分类器
        
        Args:
            data (pd.DataFrame): 训练数据
            target_col (str): 目标列名
            
        Returns:
            float: 模型评估分数
        """
        # 数据预处理
        dataset, columns = self._preprocess_data(data, target_col)
        X = dataset[columns]
        y = dataset[target_col]
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=1
        )
        
        # 平衡训练数据
        X_train_balanced, y_train_balanced = self._balance_dataset(X_train, y_train)
        
        # 训练模型
        clf = AdaBoostClassifier(random_state=0)
        clf.fit(X_train_balanced, y_train_balanced)
        
        # 预测和评估
        y_pred = clf.predict(X_test)
        return self._calculate_metrics(y_test, y_pred)