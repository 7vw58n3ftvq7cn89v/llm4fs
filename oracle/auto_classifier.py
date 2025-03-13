# src/backend_new/core/oracle/auto_classifiers.py
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from .base import BaseOracle

class AutoMLBaseOracle(BaseOracle):
    """自动机器学习的基类"""
    
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
        
        # 数据预处理
        for col in columns:
            dataset[col] = dataset[col].fillna(0)
            dataset[col] = dataset[col].replace(np.nan, 0)
            if dataset.dtypes[col] == 'object':
                dataset[col] = dataset[col].astype('category')
                dataset[col] = dataset[col].cat.codes
                
        dataset[target_col] = dataset[target_col].astype(int)
        return dataset, columns
    
    def _feature_selection(self, X_train: pd.DataFrame) -> List[str]:
        """特征选择
        
        Args:
            X_train (pd.DataFrame): 训练数据
            
        Returns:
            List[str]: 选中的特征列表
        """
        var_thr = VarianceThreshold(threshold=0.1)
        var_thr.fit(X_train)
        return [col for col in X_train.columns 
                if col not in X_train.columns[var_thr.get_support()]]
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算评估指标
        
        Args:
            y_true (np.ndarray): 真实标签
            y_pred (np.ndarray): 预测标签
            
        Returns:
            float: F1分数
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        return 2 * precision * recall / (precision + recall)


class AutoSklearnOracle(AutoMLBaseOracle):
    """使用Auto-Sklearn的分类器"""
    
    def train(self, data: pd.DataFrame, target_col: str) -> float:
        """训练Auto-Sklearn分类器
        
        Args:
            data (pd.DataFrame): 训练数据
            target_col (str): 目标列名
            
        Returns:
            float: 模型评估分数
        """
        import autosklearn.classification
        
        # 数据预处理
        dataset, columns = self._preprocess_data(data, target_col)
        X = dataset[columns]
        y = dataset[target_col]
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=1
        )
        
        # 特征选择
        excluded_cols = self._feature_selection(X_train)
        X_train = X_train.drop(excluded_cols, axis=1)
        X_test = X_test.drop(excluded_cols, axis=1)
        
        # 训练模型
        clf = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120,
            per_run_time_limit=30,
            tmp_folder="/tmp/autosklearn_parallel_1_example_tmp",
            n_jobs=4,
            memory_limit=30072,
            seed=5
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        return self._calculate_metrics(y_test, y_pred)


class TPOTOracle(AutoMLBaseOracle):
    """使用TPOT的分类器"""
    
    def train(self, data: pd.DataFrame, target_col: str) -> float:
        from tpot import TPOTClassifier
        
        # 数据预处理
        dataset, columns = self._preprocess_data(data, target_col)
        X = dataset[columns]
        y = dataset[target_col]
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=1
        )
        
        # 特征选择
        excluded_cols = self._feature_selection(X_train)
        X_train = X_train.drop(excluded_cols, axis=1)
        X_test = X_test.drop(excluded_cols, axis=1)
        
        # 训练模型
        pipeline_optimizer = TPOTClassifier(
            generations=5,
            population_size=20,
            cv=5,
            random_state=42,
            verbosity=2
        )
        
        pipeline_optimizer.fit(X_train, y_train)
        y_pred = pipeline_optimizer.predict(X_test)
        
        return self._calculate_metrics(y_test, y_pred)


class PyCaretOracle(AutoMLBaseOracle):
    """使用PyCaret的分类器"""
    
    def train(self, data: pd.DataFrame, target_col: str) -> float:
        from pycaret.classification import setup, compare_models, predict_model
        
        # 数据预处理
        dataset, columns = self._preprocess_data(data, target_col)
        
        # 准备训练数据
        train_data = dataset.copy()
        s = setup(train_data, target=target_col, silent=True)
        
        # 训练和评估
        best_model = compare_models()
        predictions = predict_model(best_model, data=train_data)
        
        return self._calculate_metrics(
            dataset[target_col],
            predictions['Label']
        )