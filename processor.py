import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
#llm
from feature_analyze.analyzer import LLMFeatureAnalyzer
# evaluate
from oracle.base import BaseOracle
from oracle.classifier import ClassifierOracle
from table_process import Table
#logger
from infrastructure.logger import get_logger
import json
import copy
from dotenv import load_dotenv
import os
load_dotenv()
logger = get_logger(__name__)


@dataclass
class ProcessorConfig:
    """处理器配置"""
    base_table_name: str = "schools"
    base_table_description: str = "2009-2010 schools performance data"
    target_col: str = "2009-2010 OVERALL GRADE"
    task_type: str = "classification"  # or "regression"
    baseline_methods: List[str] = field(
        default_factory=lambda: ['llm', 'all', 'random', 'mutual_info']
    )
    random_seed: int = 42

    result_path: str = "dsV2.5"
    
    # LLM配置
    API_KEY: str = os.getenv("SILICONFLOW_API_KEY")
    BASE_URL: str = os.environ["SILICONFLOW_API_BASE"]
    MODEL_NAME: str = "deepseek-ai/DeepSeek-V2.5"

    def __post_init__(self):
        """初始化后自动设置路径"""
        # 设置基础数据路径
        self.base_data_path = Path(f"data/base_tables/{self.base_table_name}.csv")
        
        # 设置增强数据路径
        self.augment_data_path = Path(f"data/join_datasets/{self.base_table_name}")
        
        # 设置增强数据信息路径
        self.augment_data_info_path = Path(
            f"data/join_info/{self.base_table_name}_join_info.csv"
        )
        
        # 添加结果保存路径
        self.results_dir = Path(f"results/{self.base_table_name}/{self.result_path}")
        self.log_dir = Path(f"results/{self.base_table_name}/{self.result_path}/logs")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationResult:
    """评估结果"""
    method: str          
    score: float
    improvement: float
    feature_list: List[str]
    merged_df: pd.DataFrame

@dataclass
class MethodState:
    """每种方法的状态"""
    current_table: Table
    current_score: float
    feature_history: List[str] = field(default_factory=list)
    performance_history: List[Dict] = field(default_factory=list)

class FeatureAugmentProcessor:
    def __init__(
            self, 
            config: ProcessorConfig,
            oracle: Optional[BaseOracle] = None,
        ):
        self.config = config
        self.base_df = pd.read_csv(config.base_data_path, low_memory=False)
        self.base_table = Table(
            data_path=config.base_data_path,
            dataset_name=config.base_table_name,
            description=config.base_table_description
        )
        
        # 为每种方法初始化状态
        self.method_states = {
            method: MethodState(
                current_table=copy.deepcopy(self.base_table),
                current_score=0.0
            )
            for method in config.baseline_methods
        }
        
        self.llm_analyzer = LLMFeatureAnalyzer(
            api_key=config.API_KEY, 
            base_url=config.BASE_URL, 
            model_name=config.MODEL_NAME
        )
        # 初始化评估器
        self.evaluator = oracle or ClassifierOracle(name="oracle")
        # 初始化初始分数
        self.initial_score = self.evaluator.train(
            data=self.base_table.get_df(), 
            target_col=self.config.target_col
        )

        # 加载增强表
        self.augment_table_pool = []
        self._load_augment_tables(
            config.augment_data_path, 
            pd.read_csv(config.augment_data_info_path, low_memory=False)
        )

        # 初始化每个方法的初始分数
        for method, state in self.method_states.items():
            state.current_score = self.evaluator.train(
                data=state.current_table.get_df(), 
                target_col=self.config.target_col
            )

    def _load_augment_tables(self, augment_data_path: Path, augment_data_info:pd.DataFrame):
        for _, row in augment_data_info.iterrows():
            # 创建table对象并加入table pool
            if not Path(f"{augment_data_path}/{row['dataset_name']}.csv").exists():
                continue
            new_table = Table(
                data_path=Path(f"{augment_data_path}/{row['dataset_name']}.csv"),
                score=row['score'],
                dataset_name=row['dataset_name'],
                left_key=row['left_columns_names'],
                right_key=row['right_columns_names'],
                size=row['size'],
                description=row['description']
                )
            self.augment_table_pool.append(new_table)
    
        
    def feature_selection_from_table(self, augment_table: Table, method: str = 'llm') -> List[str]:
        """从表格中提取特征
        
        Args:
            augment_table: 增强表对象
            method: 特征选择方法，支持 'llm', 'all', 'random', 'mutual_info'
            
        Returns:
            List[str]: 选择的特征列表
        """
        augment_df = augment_table.get_df()
        all_features = [col for col in augment_df.columns 
                       if col != augment_table.right_key]
        
        if method == 'llm':
            suggestion = self.llm_analyzer.analyze_features(
                base_table=self.method_states[method].current_table,
                augment_table=augment_table, 
                target_attribute=self.config.target_col,
                table_analysis=False
            )
            feature_list = suggestion['recommended_features']
        
        elif method == 'all':
            feature_list = all_features
            
        elif method == 'random':
            k = len(all_features) // 2  # 默认选择一半特征
            np.random.seed(self.config.random_seed)
            feature_list = list(np.random.choice(all_features, k, replace=False))
            
        elif method == 'mutual_info':
            k = len(all_features) // 2  # 默认选择一半特征
            feature_list = self._select_features_by_mutual_info(
                augment_table=augment_table,
                n_features=k,
                base_df=self.method_states[method].current_table.get_df()
            )
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")

        # logger.info(f"Method {method} selected features:")
        # for feature in feature_list:
        #     logger.info(feature)
        return feature_list
    
    def _table_selection(self)-> Table:
        """从table_pool中选择一个增强表"""
        # TODO: 优化采样方法，例如依据增强表的大小、特征数量等因素进行采样
        
        sample_table = self.augment_table_pool.pop()
        logger.info(f"Sampled table: {sample_table.name}")
        return sample_table
    
    def _save_round_results(self, round_num: int, augment_table: Table, round_results: List[Dict]):
        """保存每轮的结果到文件
        
        Args:
            round_num: 轮次编号
            augment_table: 当前使用的增强表
            round_results: 该轮各方法的结果列表
        """
        log_file = self.config.log_dir / f"round_{round_num}.json"
        
        round_log_data = {
            "round": round_num,
            "augment_table": augment_table.name,
            "augment_table_size": augment_table.size,
            "timestamp": pd.Timestamp.now().isoformat(),
            "methods_results": round_results
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(round_log_data, f, indent=2, ensure_ascii=False)
            
        logger.debug(f"Round {round_num} results saved to {log_file}")

    def run(self):
        """主函数"""
        round_num = 0
        
        while not self.is_done():
            round_num += 1
            try:
                augment_table = self._table_selection()
                logger.info(f"\nRound {round_num}: Processing table {augment_table.name}")
                
                round_results = []  # 存储当前轮次的所有结果
                
                # 对每种方法进行特征选择和评估
                for method in self.config.baseline_methods:
                    try:
                        state = self.method_states[method]
                        
                        selected_features = self.feature_selection_from_table(
                            augment_table=augment_table,
                            method=method
                        )
                        
                        evaluation = self._evaluate_features(
                            augment_table=augment_table, 
                            feature_list=selected_features,
                            current_df=state.current_table.get_df(),
                            current_score=state.current_score
                        )
                        
                        # 记录当前方法的结果
                        method_result = {
                            "method": method,
                            "selected_features": selected_features,
                            "num_features": len(selected_features),
                            "score": float(evaluation['score']),
                            "improvement": float(evaluation['improvement']),
                            "current_features_num": len(state.current_table.get_df().columns),
                            "current_features": state.current_table.get_df().columns.tolist()
                        }
                        round_results.append(method_result)
                        
                        if evaluation['improvement'] > 0:
                            self._update_method_state(
                                method=method,
                                evaluation=evaluation,
                                selected_features=selected_features,
                                round_num=round_num
                            )
                            
                        logger.info(
                            f"Method {method:12} | Features: {len(selected_features):3d} | "
                            f"Score: {evaluation['score']:.4f} | "
                            f"Improvement: {evaluation['improvement']:+.4f}"
                        )
                        
                    except Exception as e:
                        logger.error(f"Method {method} failed: {str(e)}")
                        round_results.append({
                            "method": method,
                            "error": str(e)
                        })
                
                # 保存当前轮次的结果
                self._save_round_results(round_num, augment_table, round_results)
                
                # 每轮结束后输出各方法的当前状态
                self._report_current_status()
                
            except Exception as e:
                logger.error(f"Round {round_num} failed: {str(e)}")
        
        # 输出最终对比报告并保存到文件
        self._report_final_comparison()
        return self.method_states

    def _evaluate_features(self, 
                         augment_table: Table, 
                         feature_list: List[str],
                         current_df: pd.DataFrame,
                         current_score: float) -> Dict:
        """评估特征有效性"""
        if not feature_list:
            return {
                'merged_df': current_df,
                'score': current_score,
                'improvement': 0,
                'feature_list': []
            }
        
        merged_df = augment_table.join_features(current_df, feature_list)

        score = self.evaluator.train(merged_df, self.config.target_col)
        improvement = score - current_score

        return {
            'merged_df': merged_df,
            'score': score,
            'improvement': improvement,
            'feature_list': feature_list
        }

    def _update_method_state(self, 
                           method: str, 
                           evaluation: Dict,
                           selected_features: List[str],
                           round_num: int):
        """更新方法状态"""
        state = self.method_states[method]
        state.current_table.df = evaluation['merged_df']
        state.current_score = evaluation['score']
        state.feature_history.extend(selected_features)
        
        # 记录性能历史
        state.performance_history.append({
            'round': round_num,
            'features_added': selected_features,
            'score': evaluation['score'],
            'improvement': evaluation['improvement']
        })

    def _report_current_status(self):
        """报告当前各方法状态"""
        logger.info("\nCurrent Status Summary:")
        for method, state in self.method_states.items():
            logger.info(
                f"Method {method:10} | "
                f"Features: {len(state.current_table.get_df().columns):3d} | "
                f"Score: {state.current_score:.4f}"
            )

    def _report_final_comparison(self):
        """生成最终对比报告并保存到文件"""
        logger.info("\nFinal Performance Comparison:")
        
        # 计算每个方法的总体改进
        results = []
        for method, state in self.method_states.items():
            total_improvement = state.current_score - self.initial_score
            
            results.append({
                'method': method,
                'final_score': state.current_score,
                'total_improvement': total_improvement,
                'features_added': len(state.feature_history),
                'rounds_improved': len(state.performance_history),
                'feature_history': state.feature_history,
                'performance_history': state.performance_history
            })
        
        # 转换为DataFrame并排序
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('total_improvement', ascending=False)
        
        # 记录到日志
        logger.info("\n" + str(df_results))
        
        # 准备详细的改进历史
        detailed_history = {}
        for method, state in self.method_states.items():
            detailed_history[method] = {
                'history': state.performance_history,
                'final_score': state.current_score,
                'total_improvement': state.current_score - self.initial_score,
                'features_added': state.feature_history
            }
            
            logger.info(f"\nMethod: {method}")
            for record in state.performance_history:
                logger.info(
                    f"Round {record['round']}: "
                    f"Score {record['score']:.4f} "
                    f"(+{record['improvement']:.4f})"
                )
        
        # 保存最终结果
        final_results = {
            'summary': df_results.to_dict(orient='records'),
            'detailed_history': detailed_history,
            'config': {
                'base_table': self.config.base_table_name,
                'target_col': self.config.target_col,
                'task_type': self.config.task_type,
                'baseline_methods': self.config.baseline_methods,
                'random_seed': self.config.random_seed
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # 保存为JSON文件
        results_file = self.config.results_dir / 'final_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n最终结果已保存至：{results_file}")
        
        # 保存为CSV文件（仅summary部分）
        csv_file = self.config.results_dir / 'final_results_summary.csv'
        df_results.to_csv(csv_file, index=False)
        logger.info(f"结果摘要已保存至：{csv_file}")

    def is_done(self):
        """判断是否完成"""
        return len(self.augment_table_pool) == 0

    def _select_features_by_mutual_info(
            self, 
            augment_table: Table, 
            n_features: int,
            base_df: pd.DataFrame
        ) -> List[str]:
        """使用互信息方法选择特征"""
        try:
            # 准备数据
            base_df = base_df[[augment_table.left_key, self.config.target_col]].copy()
            augment_df = augment_table.get_df()
            all_features = [col for col in augment_df.columns 
                           if col != augment_table.right_key]
            
            # 合并数据集
            merged = base_df.merge(
                augment_df, 
                left_on=augment_table.left_key,
                right_on=augment_table.right_key,
                how='left'
            )
            
            # 预处理特征和目标变量
            X = merged[all_features].copy()
            y = merged[self.config.target_col].copy()
            
            # 处理目标变量
            if self.config.task_type == 'classification':
                y = y.astype('category').cat.codes
            else:
                y = pd.to_numeric(y, errors='coerce')
                y = y.fillna(y.median())
            
            # 处理特征
            X_processed = pd.DataFrame()
            for col in all_features:
                try:
                    series = X[col]
                    # 检查是否为数值型
                    if pd.api.types.is_numeric_dtype(series):
                        # 数值型数据处理
                        processed = pd.to_numeric(series, errors='coerce')
                        processed = processed.fillna(processed.median() if not processed.isnull().all() else 0)
                    else:
                        # 类别型数据处理
                        processed = series.fillna('MISSING')
                        processed = processed.astype('category').cat.codes
                    
                    X_processed[col] = processed
                    logger.debug(f"特征 {col} 处理后类型: {X_processed[col].dtype}")
                    
                except Exception as e:
                    logger.warning(f"处理特征 {col} 时出错: {str(e)}")
                    X_processed[col] = 0
            
            # 验证数据类型
            non_numeric_cols = [col for col in X_processed.columns 
                              if not pd.api.types.is_numeric_dtype(X_processed[col])]
            
            if non_numeric_cols:
                logger.error(f"以下特征仍为非数值类型: {non_numeric_cols}")
                logger.debug(f"特征类型详情:\n{X_processed.dtypes}")
                raise ValueError(f"特征 {non_numeric_cols} 无法转换为数值类型")
            
            # 确保所有值都是有限数
            X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
            X_processed = X_processed.fillna(0)
            
            # 计算互信息分数
            if self.config.task_type == 'classification':
                mi_scores = mutual_info_classif(X_processed, y)
            else:
                mi_scores = mutual_info_regression(X_processed, y)
            
            # 选择特征
            feature_scores = list(zip(all_features, mi_scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in feature_scores[:n_features]]
            
            # 记录特征分数和类型信息
            score_dict = {f: score 
                         for f, score in feature_scores}
            logger.debug(f"互信息特征分数: {score_dict}")
            logger.debug(f"特征类型:\n{X_processed.dtypes}")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"互信息特征选择失败: {str(e)}")
            logger.debug(f"目标变量类型: {y.dtype}")
            logger.debug(f"原始特征类型:\n{X.dtypes}")
            raise ValueError(f"互信息特征选择失败: {str(e)}")




if __name__ == '__main__':
    BASE_TABLE_NAME = 'aqe-nta'
    api_key = os.getenv("SILICONFLOW_API_KEY")
    base_url = os.getenv("SILICONFLOW_API_BASE")
    model_name = "deepseek-ai/DeepSeek-V2.5"

    base_tables_info = pd.read_csv('data/base_table_info.csv')
    row = base_tables_info[base_tables_info['base_table_name'] == BASE_TABLE_NAME]

    config = ProcessorConfig(
        base_table_name=BASE_TABLE_NAME,
        base_table_description=row['description'].values[0],
        target_col=row['target_attribute'].values[0],
        baseline_methods=['llm', 'all', 'random', 'mutual_info'],
        random_seed=42,
        result_path="dsV2.5_with_analysis",
        API_KEY=api_key,
        BASE_URL=base_url,
        MODEL_NAME=model_name
    )
    processor = FeatureAugmentProcessor(config=config)
    method_states = processor.run()

    # 获取最佳方法的结果
    best_method = max(method_states.items(), key=lambda x: x[1].current_score)
    print(f"Best method: {best_method[0]}")
    print(f"Best score: {best_method[1].current_score}")


"""
aqe-nta : PM_tertiles
schools : 2009-2010 OVERALL GRADE

"""