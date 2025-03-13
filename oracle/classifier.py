# src/backend_new/core/oracle/classifier.py
import copy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from typing import List, Tuple
from .base import BaseOracle

class ClassifierOracle(BaseOracle):

    def _preprocess_data(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
        """预处理数据，包括处理缺失值和类别型数据。

        Args:
            data (pd.DataFrame): 原始数据
            target_col (str): 目标列名

        Returns:
            Tuple[pd.DataFrame, List[str]]: 处理后的数据集和特征列名列表
        """
        dataset = copy.deepcopy(data)
        columns = list(dataset.columns)
        # print(f"columns: {columns}")
        # print(f"target_col: {target_col}")
        columns.remove(target_col)

        # 删除目标列为空值的行
        dataset = dataset[dataset[target_col].notna()]

        # 处理每一列的数据
        for col in columns:
            if dataset[col].dtype.name == 'object':
                # 对于类别型数据，先填充缺失值，再进行编码
                dataset[col] = dataset[col].fillna('MISSING')
                dataset[col] = dataset[col].astype('category')
                dataset[col] = dataset[col].cat.codes
            else:
                # 对于数值型数据，用中位数填充缺失值
                # 如果全为空值，则填充为0
                if dataset[col].isnull().all():
                    dataset[col] = 0
                else:
                    dataset[col] = dataset[col].fillna(dataset[col].median())
        

        # 最后检查确保没有任何 NaN 值
        assert not dataset.isnull().any().any(), "数据中仍然存在 NaN 值"

        return dataset, columns

    def _feature_selection(self, X_train: pd.DataFrame) -> List[str]:
        """使用方差阈值进行特征选择。
        Args:
            X_train (pd.DataFrame): 训练数据特征
        Returns:
            List[str]: 被选中的特征列名列表
        """
        var_thr = VarianceThreshold(threshold=0.1)
        var_thr.fit(X_train)
        return [col for col in X_train.columns 
                if col not in X_train.columns[var_thr.get_support()]]

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算模型评估指标（F1分数）。
        
        Returns:
            float: F1分数
        """
        return f1_score(y_true, y_pred, average='macro')

    def train(self, data: pd.DataFrame, target_col: str) -> float:
        """训练随机森林分类器并返回F1分数。

        Args:
            data (pd.DataFrame): 训练数据
            target_col (str): 目标列名

        Returns:
            float: 平均F1分数
        """
        # 数据预处理
        dataset, columns = self._preprocess_data(data, target_col)
        
        X = dataset[columns]
        y = dataset[target_col]

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        total_score = 0
        seeds = [42]  # 可配置的随机种子列表

        for seed in seeds:
            # # 特征选择
            # excluded_cols = self._feature_selection(X_train)
            # X_train_selected = X_train.drop(excluded_cols, axis=1)
            # X_test_selected = X_test.drop(excluded_cols, axis=1)

            # 训练模型
            classifier = RandomForestClassifier(random_state=seed)
            classifier.fit(X_train, y_train)

            # 预测和评估
            y_pred = classifier.predict(X_test)
            total_score += self._calculate_metrics(y_test.values, y_pred)

        return total_score / len(seeds)