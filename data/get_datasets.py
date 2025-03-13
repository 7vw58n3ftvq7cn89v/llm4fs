import requests
import json
import pandas as pd
import re
import os
from pathlib import Path
import glob
import logging
from typing import List, Dict, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetConfig:
    """数据集配置类"""
    BASE_URL = 'https://auctus.vida-nyu.org/api/v1'
    METADATA_URL = 'https://auctus.vida-nyu.org/api/v1/metadata'
    KEY_KEYWORDS = ['id', 'name', 'code', 'number', 'no', 'key']
    EXCLUDE_WORDS = ['text']
    MIN_UNIQUE_RATIO = 0.4
    MIN_SMALL_UNIQUE_RATIO = 0.2
    MIN_NOTNULL_RATIO = 0.8
    MIN_SCORE = 0.5
    MAX_SIZE = 100_000_000
    MIN_SIZE = 20_000

class ColumnAnalyzer:
    """列分析器，用于识别可能的连接键"""
    
    @staticmethod
    def identify_join_columns(dataset_path: str, threshold: float = DatasetConfig.MIN_UNIQUE_RATIO) -> List[int]:
        df = pd.read_csv(dataset_path)
        potential_keys = []
        
        for idx, col in enumerate(df.columns):
            if df[col].isna().all():
                continue
                
            unique_ratio = df[col].nunique() / len(df[col].dropna())
            col_lower = col.lower()
            
            has_key_keyword = any(keyword in col_lower for keyword in DatasetConfig.KEY_KEYWORDS)
            has_exclude_word = any(exclude_word in col_lower for exclude_word in DatasetConfig.EXCLUDE_WORDS)
            
            if (ColumnAnalyzer._check_column_validity(unique_ratio, has_key_keyword, df[col], has_exclude_word)):
                potential_keys.append(idx)
                logger.info(f"发现可能的连接键: {col} (列序号: {idx})")
                logger.info(f"  - 唯一值比例: {unique_ratio:.2%}")
                logger.info(f"  - 非空值比例: {(df[col].notna().sum() / len(df)):.2%}")
        
        return potential_keys

    @staticmethod
    def _check_column_validity(unique_ratio: float, has_key_keyword: bool, 
                             column: pd.Series, has_exclude_word: bool) -> bool:
        return ((unique_ratio >= DatasetConfig.MIN_UNIQUE_RATIO) or 
                (has_key_keyword and unique_ratio >= DatasetConfig.MIN_SMALL_UNIQUE_RATIO)) and \
               (column.notna().sum() / len(column) >= DatasetConfig.MIN_NOTNULL_RATIO) and \
               not has_exclude_word

class DatasetFetcher:
    """数据集获取器，处理与API的交互"""
    
    @staticmethod
    def query_single_key(dataset_path: str, key: int, keywords: str = '') -> List[Dict]:
        logger.info(f"尝试使用连接键 {key}")
        query_json = json.dumps({
            "keywords": keywords,
            "variables": [{"type": "tabular_variable", "columns": [key]}]
        }).encode('utf-8')
        
        try:
            with open(dataset_path, 'rb') as f:
                response = requests.post(
                    f'{DatasetConfig.BASE_URL}/search',
                    files={'query': query_json, 'data': f}
                )
                response.raise_for_status()
                return DatasetFetcher._process_query_results(response.json(), key)
        except Exception as e:
            logger.error(f"连接键 {key} 查询出错: {str(e)}")
            return []

    @staticmethod
    def download_dataset(dataset_id: str) -> pd.DataFrame:
        logger.info(f'downloading {dataset_id}...')
        try:
            response = requests.get(f'{DatasetConfig.BASE_URL}/download/{dataset_id}')
            response.raise_for_status()
            
            # 尝试不同的解析参数
            try:
                return pd.read_csv(pd.io.common.BytesIO(response.content))
            except pd.errors.ParserError:
                # 如果默认解析失败，尝试使用更宽松的解析参数
                return pd.read_csv(
                    pd.io.common.BytesIO(response.content),
                    error_bad_lines=False,  # 跳过有问题的行
                    warn_bad_lines=True,    # 显示警告信息
                    on_bad_lines='skip'     # 跳过错误行
                )
        except Exception as e:
            logger.error(f"下载数据集 {dataset_id} 时出错: {str(e)}")
            return pd.DataFrame()  # 返回空数据框而不是抛出异常

    @staticmethod
    def get_dataset_description(dataset_id: str) -> str:
        response = requests.get(f'{DatasetConfig.BASE_URL}/metadata/{dataset_id}')
        return response.json().get('metadata', {}).get('description', 'N/A')

    @staticmethod
    def _process_query_results(response_data: Dict, key: int) -> List[Dict]:
        join_datasets = []
        for result in response_data['results']:
            if result['augmentation']['type'] != 'join':
                continue
            dataset = {
                'id': result['id'],
                'dataset_name': result['metadata']['name'].replace(' ', '_').replace('/', '_'),
                'score': result['score'],
                'left_columns_names': result['augmentation']['left_columns_names'],
                'right_columns_names': result['augmentation']['right_columns_names'],
                'size': result['metadata']['size'],
                'join_key': key,
                'description': result['metadata'].get('description', 'N/A')
            }
            join_datasets.append(dataset)
            logger.info(f"找到数据集: {dataset['dataset_name']}, score:{dataset['score']}, size:{dataset['size']}")
        return join_datasets

class DatasetManager:
    """数据集管理器，处理数据集的查询和下载"""
    
    @staticmethod
    def get_joinable_datasets(dataset_path: str, keywords: str = '',query_keys: List[int] = []) -> pd.DataFrame:
        base_table_name = Path(dataset_path).stem
        potential_keys = query_keys if query_keys else ColumnAnalyzer.identify_join_columns(dataset_path)
        
        if not potential_keys:
            logger.warning("未找到合适的连接键")
            return pd.DataFrame()
            
        all_join_datasets = []
        for key in potential_keys:
            join_datasets = DatasetFetcher.query_single_key(dataset_path, key, keywords)
            all_join_datasets.extend(join_datasets)
            
        return DatasetManager._process_join_datasets(all_join_datasets)

    @staticmethod
    def download_join_datasets(base_table_name: str, join_info_path: Optional[Path] = None) -> None:
        if join_info_path is None:
            join_info_path = Path(f'join_info/{base_table_name}_join_info.csv')
            
        join_datasets = pd.read_csv(join_info_path)
        save_dir = Path(f'join_datasets/{base_table_name}')
        os.makedirs(save_dir, exist_ok=True)
        
        for _, row in join_datasets.iterrows():
            if row['size'] > DatasetConfig.MAX_SIZE or row['score'] < DatasetConfig.MIN_SCORE or row['size'] < DatasetConfig.MIN_SIZE:
                logger.info(f"跳过数据集 {row['dataset_name']} (size或score不符合要求)")
                continue
            
            if f"{row['dataset_name']}.csv" in os.listdir(save_dir):
                logger.info(f"数据集 {row['dataset_name']} 已存在")
                continue
            
            dataset = DatasetFetcher.download_dataset(row['id'])
            if dataset.empty:
                logger.warning(f"数据集 {row['dataset_name']} 下载失败")
                continue
            dataset.to_csv(save_dir / f"{row['dataset_name']}.csv", index=False)
            logger.info(f"已保存数据集 {row['dataset_name']}, 行数: {len(dataset)}")

    @staticmethod
    def _process_join_datasets(all_join_datasets: List[Dict]) -> pd.DataFrame:
        if not all_join_datasets:
            logger.warning("未找到任何可连接的数据集")
            return pd.DataFrame()
            
        df_join_datasets = pd.DataFrame(all_join_datasets)
        df_join_datasets['left_columns_names'] = df_join_datasets['left_columns_names'].apply(
            lambda x: re.sub(r'[\[\]\'"]+', '', str(x)))
        df_join_datasets['right_columns_names'] = df_join_datasets['right_columns_names'].apply(
            lambda x: re.sub(r'[\[\]\'"]+', '', str(x)))
            
        df_join_datasets = df_join_datasets.drop_duplicates(subset=['id'])
        logger.info(f"总共找到 {len(df_join_datasets)} 个唯一的可连接数据集")
        return df_join_datasets

def main():
    """主函数"""
    os.makedirs('base_tables', exist_ok=True)
    os.makedirs('join_info', exist_ok=True)
    os.makedirs('join_datasets', exist_ok=True)
    
    # 示例用法
    BASE_TABLE_NAME = 'Poverty'
    process_single_table(BASE_TABLE_NAME, download=True)

def process_single_table(base_table_name: str, download: bool = False) -> None:
    """处理单个基表"""
    base_table_info = DatasetManager.get_joinable_datasets(f'base_tables/{base_table_name}.csv')
    
    output_path = Path(f'join_info/{base_table_name}_join_info.csv')
    if output_path.exists():
        old_info = pd.read_csv(output_path)
        base_table_info = pd.concat([old_info, base_table_info], ignore_index=True)
        base_table_info = base_table_info.drop_duplicates(subset=['id'])
    
    base_table_info.to_csv(output_path, index=False)
    logger.info(f'连接信息已保存至 {output_path}')
    
    if download:
        DatasetManager.download_join_datasets(base_table_name, output_path)

if __name__ == '__main__':
    main()
