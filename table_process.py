import pandas as pd
from typing import List
# import numpy as np

from pathlib import Path

#logger
from infrastructure.logger import get_logger

logger = get_logger(__name__)

class Table:
    def __init__(
            self, 
            data_path: str,
            dataset_name: str, 
            score: float=None,
            left_key: str=None, 
            right_key: str=None,
            size: int=None,
            description: str=None,
            target_attribute: str=None
        ):
        self.df = pd.read_csv(Path(f"{data_path}")) # 为了减小内存使用，可以先不加载
        self.score = score
        self.name = dataset_name
        self.left_key = left_key
        self.right_key = right_key
        self.size = size
        self.description = description
        self.target_attribute = target_attribute
        self.schema_info_cache = {}

    def get_df(self, feature_list: List[str]=[]) -> pd.DataFrame:
        """获取数据框，支持原始特征名和重命名后的特征名
        
        Args:
            feature_list: 要获取的特征列表，可以是原始名或重命名后的名称
            
        Returns:
            pd.DataFrame: 包含指定特征的数据框
        """
        if not feature_list:
            return self.df
        
        # 确保连接键在结果中
        if self.right_key not in feature_list:
            feature_list.append(self.right_key)
        
        feature_list = [f for f in feature_list if f in self.df.columns]
        
        return self.df[feature_list]
    
    def get_columns(self) -> List[str]:
        """获取列名列表"""
        return self.df.columns.tolist()
    
    def get_table_info(self, schema: bool=True, description: bool=True):
        """获取表格描述，放到提示词里"""
        description_text = ""
        schema_text = ""
        if schema:
            schema_text = "Schema Information:\n"
            for col_name, col in self.df.items():
                # check cache
                if col_name in self.schema_info_cache:
                    single_schema_text = self.schema_info_cache[col_name]
                else:
                    # calculate new schema info
                    if col.dtype != 'object' and col.dtype != str:
                        single_schema_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "min": {col.min()}, "max": {col.max()}}}'
                    else:
                        most_freq_vals = col.value_counts().index.tolist()
                        example_cells = most_freq_vals[:min(3, len(most_freq_vals))]
                        single_schema_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}","unique_count": {col.nunique()}, "cell_examples": {example_cells}}}'
                    # update cache
                    self.schema_info_cache[col_name] = single_schema_text
                schema_text += single_schema_text
        
        if description:
            description_text = f"Table name: {self.name}\nTable description: {self.description}"
        
        text = description_text + schema_text
        
        return text
    
    def join_features(self, df: pd.DataFrame, feature_list: List[str]):
        """将特征列表中的特征连接到表中，避免一对多连接导致的行数扩充"""
        # 列名重复的处理：直接删除
        feature_list = [col for col in feature_list if col not in df.columns]
        
        if self.right_key not in feature_list:
            feature_list.append(self.right_key)
        
        # 获取子集
        subset_df = self.get_df(feature_list)
        
        # 检查是否存在一对多关系
        right_key_counts = subset_df[self.right_key].value_counts()
        has_many = (right_key_counts > 1).any()
        
        if has_many:
            # logger.info("检测到一对多关系，对特征进行聚合处理")
            
            def safe_agg(series):
                """安全的聚合函数
                
                Args:
                    series: pandas.Series, 需要聚合的列
                Returns:
                    聚合后的值
                """
                if series.empty:
                    return None
                    
                # 处理全为空值的情况
                if series.isna().all():
                    return None
                    
                # 如果所有非空值都相同，直接返回该值
                unique_values = series.dropna().unique()
                if len(unique_values) == 1:
                    return unique_values[0]
                
                # 根据数据类型选择聚合方法
                dtype = series.dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    # 数值型取平均值
                    return series.mean()
                else:
                    # 非数值型取出现次数最多的值
                    try:
                        # 将所有值转换为字符串后再计算众数
                        str_series = series.astype(str)
                        mode_result = str_series.mode()
                        return mode_result.iloc[0] if not mode_result.empty else None
                    except Exception as e:
                        logger.warning(f"计算众数时出错: {str(e)}")
                        return None
            
            # 对非连接键的列进行聚合
            agg_dict = {
                col: safe_agg 
                for col in subset_df.columns 
                if col != self.right_key
            }
            
            # 进行聚合
            subset_df = subset_df.groupby(self.right_key).agg(agg_dict)
            
            # 重置索引，使连接键成为列
            subset_df = subset_df.reset_index()
            
            # 打印聚合后的统计信息
            for col in subset_df.columns:
                if col != self.right_key:
                    null_count = subset_df[col].isnull().sum()
                    unique_count = subset_df[col].nunique()
                    logger.debug(f"列 {col}: 空值数量={null_count}, 唯一值数量={unique_count}")
        
        # 进行左连接
        result_df = df.merge(
            subset_df, 
            left_on=self.left_key, 
            right_on=self.right_key, 
            how='left'
        )
        
        # 如果连接键不同，删除右表的连接键
        if self.left_key != self.right_key:
            result_df = result_df.drop(columns=[self.right_key])
        
        # 验证行数是否保持不变
        assert len(result_df) == len(df), f"连接后行数发生变化: {len(df)} -> {len(result_df)}"
        
        return result_df


class BaseTable(Table):
    """基表，需要保留df"""
    def __init__(
            self, 
            data_path: str,
            dataset_name: str,  
            size: int=None,
            description: str=None,
            target_attribute: str=None
        ):
        super().__init__(
            data_path=data_path,
            dataset_name=dataset_name, 
            size=size,
            description=description,
            target_attribute=target_attribute
        )
        self.df = pd.read_csv(Path(f"{data_path}"))
        self.target_attribute = target_attribute
        self.task_type = "Classification"

class AugmentTable(Table):
    """增强表，不需要保留df"""
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    base_table_name = "aqe-nta"
    augment_table_name = "Buildings_Selected_for_the_Alternative_Enforcement_Program_(AEP)"
    df_join_info = pd.read_csv(f"data/join_info/{base_table_name}_join_info.csv")
    df_join_info = df_join_info[df_join_info["dataset_name"] == augment_table_name]
    # 读取基础表
    base_df = pd.read_csv(f"data/base_tables/{base_table_name}.csv")

    # 读取增强表
    table = Table(
        data_path=f"data/join_datasets/{base_table_name}/{augment_table_name}.csv", 
        score=0.8, 
        dataset_name=augment_table_name, 
        left_key=df_join_info["left_columns_names"].values[0], 
        right_key=df_join_info["right_columns_names"].values[0], 
        size=df_join_info["size"].values[0],
        description=df_join_info["description"].values[0]
    )
    print(f"Augment table columns: {table.get_columns()}")
    augment_df = table.join_features(base_df, table.get_columns())
    print(f"Base table columns: {base_df.columns.tolist()}, shape: {base_df.shape}")
    print(f"Augmented base table columns: {augment_df.columns.tolist()}, shape: {augment_df.shape}")
