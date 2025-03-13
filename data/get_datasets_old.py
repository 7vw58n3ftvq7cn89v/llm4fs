import requests
import json
import pandas as pd
import re
import os
from pathlib import Path
import glob

"""
usage: 把基表放到base_tables目录下，运行脚本，会自动查询可连接的数据集，并下载到join_datasets目录下
"""

def search_datasets(keywords):
    response = requests.post(
        'https://auctus.vida-nyu.org/api/v1/search',
        json={'keywords': keywords},
    )
    response.raise_for_status()
    for result in response.json()['results']:
        print(result['id'])

def identify_join_columns(dataset_path, threshold=0.8):
    """
    自动识别可能的连接键
    Args:
        dataset_path: 数据集路径
        threshold: 唯一值比例阈值
    Returns:
        potential_keys: 可能的连接键列序号列表
    """
    df = pd.read_csv(dataset_path)
    potential_keys = []

    small_threshold = 0.2
    
    # 关键词列表
    key_keywords = ['id', 'name', 'code', 'number', 'no', 'key']

    # 排除的列名
    exclude_words = ['text']
    
    for idx, col in enumerate(df.columns):
        # 跳过全是空值的列
        if df[col].isna().all():
            continue
            
        # 计算非空唯一值比例
        unique_ratio = df[col].nunique() / len(df[col].dropna())
        
        # 检查列名是否包含关键词
        col_lower = col.lower()
        has_key_keyword = any(keyword in col_lower for keyword in key_keywords)
        has_exclude_word = any(exclude_word in col_lower for exclude_word in exclude_words)
        
        # 条件：
        # 1. 唯一值比例高于阈值
        # 2. 或列名包含关键词
        # 3. 且非空值比例高于80%
        # 4. 且非排除列
        if ((unique_ratio >= threshold) or (has_key_keyword and unique_ratio >= small_threshold)) and \
           (df[col].notna().sum() / len(df) >= 0.8) and \
           not has_exclude_word:
            potential_keys.append(idx)
            print(f"发现可能的连接键: {col} (列序号: {idx})")
            print(f"  - 唯一值比例: {unique_ratio:.2%}")
            print(f"  - 非空值比例: {(df[col].notna().sum() / len(df)):.2%}")
    
    return potential_keys

def query_single_key(dataset_path, key, keywords=''):
    """
    使用单个连接键查询可连接的数据集
    Args:
        dataset_path: 数据集路径
        key: 连接键列序号
        keywords: 搜索关键词
    Returns:
        list: 查询到的数据集列表
    """
    print(f"尝试使用连接键 {key}")
    query_json = json.dumps({
        "keywords": keywords,
        "variables": [
            {
                "type": "tabular_variable",
                "columns": [key]
            }
        ]
    }).encode('utf-8')
    
    join_datasets = []
    try:
        with open(dataset_path, 'rb') as f:
            response = requests.post(
                'https://auctus.vida-nyu.org/api/v1/search',
                files={
                    'query': query_json,
                    'data': f,
                },
            )
            
            if response.status_code != 200:
                print(f"连接键 {key} 查询失败，状态码: {response.status_code}")
                return join_datasets
            
            for result in response.json()['results']:
                if result['augmentation']['type'] != 'join':
                    continue
                dataset = {
                    'id': result['id'], 
                    'dataset_name': result['metadata']['name'].replace(' ', '_'),
                    'score': result['score'],
                    'left_columns_names': result['augmentation']['left_columns_names'], 
                    'right_columns_names': result['augmentation']['right_columns_names'],
                    'size': result['metadata']['size'],
                    'join_key': key,
                    'description': result['metadata']['description']
                }
                join_datasets.append(dataset)
                print(f"使用连接键 {key} 找到数据集: {dataset['dataset_name']}, score:{dataset['score']}, size:{dataset['size']}")
                
    except Exception as e:
        print(f"连接键 {key} 查询出错: {str(e)}")
    
    return join_datasets

def get_joinable_datasets(dataset_path, keywords=''):
    """
    查询所有可能的连接键，获取可连接的数据集
    Args:
        dataset_path: 数据集路径
        keywords: 搜索关键词
    Returns:
        DataFrame: 所有可连接的数据集信息
    """
    base_table_name = Path(dataset_path).stem
    
    # 自动识别可能的连接键
    potential_keys = identify_join_columns(dataset_path)
    print(f"table:{base_table_name}, potential_keys: {potential_keys}")
    if not potential_keys:
        print("警告：未找到合适的连接键")
        return pd.DataFrame()
    
    # 收集所有连接键的查询结果
    all_join_datasets = []
    for key in potential_keys:
        join_datasets = query_single_key(dataset_path, key, keywords)
        all_join_datasets.extend(join_datasets)
    
    # 如果没有找到任何结果
    if not all_join_datasets:
        print("未找到任何可连接的数据集")
        return pd.DataFrame()
    
    # 合并所有结果到一个DataFrame
    df_join_datasets = pd.DataFrame(all_join_datasets)
    
    # 清理列名
    def clean_column_name(column_name):
        if not isinstance(column_name, str):
            column_name = str(column_name)
        return re.sub(r'[\[\]\'"]+', '', column_name)
    
    df_join_datasets['left_columns_names'] = df_join_datasets['left_columns_names'].apply(clean_column_name)
    df_join_datasets['right_columns_names'] = df_join_datasets['right_columns_names'].apply(clean_column_name)
    
    # 去除重复的数据集
    df_join_datasets = df_join_datasets.drop_duplicates(subset=['id'])
    
    print(f"总共找到 {len(df_join_datasets)} 个唯一的可连接数据集")
    return df_join_datasets

def download_dataset(dataset_id):
    print(f'downloading {dataset_id}...')
    response = requests.get(
        f'https://auctus.vida-nyu.org/api/v1/download/{dataset_id}',
    )
    
    # 直接从内存中的字节数据读取
    table = pd.read_csv(pd.io.common.BytesIO(response.content))
    return table
    # table.to_csv(f'join_datasets/{dataset_id}.csv', index=False)

def get_and_check_dataset(dataset_id):
    response = requests.get(
        f'https://auctus.vida-nyu.org/api/v1/download/{dataset_id}',
    )

    # 1. 检查响应状态码
    print(f"状态码: {response.status_code}")

    # 2. 检查响应头
    print("\n响应头:")
    print(response.headers)

    # 3. 检查内容类型
    print(f"\n内容类型: {response.headers.get('content-type')}")

    # 4. 尝试解析内容
    try:
        # 如果是JSON格式
        if 'application/json' in response.headers.get('content-type', ''):
            data = response.json()
            print("\nJSON数据:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        # 如果是CSV格式
        elif 'text/csv' in response.headers.get('content-type', ''):
            df = pd.read_csv(pd.io.common.BytesIO(response.content))
            print("\nCSV数据预览:")
            print(df.head())
            print(f"\n数据形状: {df.shape}")
            print("\n列名:")
            print(df.columns.tolist())
        # 如果是文本格式
        else:
            print("\n原始内容预览:")
            print(response.text[:500])  # 只显示前500个字符
    except Exception as e:
        print(f"解析内容时出错: {str(e)}")

def download_join_datasets(base_table_name, join_info_path=None):
    """
    下载增强表
    """
    if join_info_path is None:
        join_info_path = Path(f'join_info/{base_table_name}_join_info.csv')
    join_datasets = pd.read_csv(join_info_path)
    # 为每个基表创建独立的保存目录
    save_dir = Path(f'join_datasets/{base_table_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    for _, row in join_datasets.iterrows():
        if row['size'] > 100_000_000 or row['score'] < 0.5:
            print(f"skip dataset {row['dataset_name']} because of size or score")
            continue
        dataset = download_dataset(row['id'])
        print(f"dataset {row['dataset_name']} length: {len(dataset)}")
        dataset.to_csv(Path(f'{save_dir}/{row["dataset_name"]}.csv'), index=False)


def process_base_tables(query=True, download=False):
    """处理base_tables文件夹中的所有基表"""
    # 创建必要的目录
    os.makedirs('base_tables', exist_ok=True)
    os.makedirs('join_info', exist_ok=True)
    os.makedirs('join_datasets', exist_ok=True)
    
    # 获取所有基表
    base_tables = glob.glob('base_tables/*.csv')

    # 1. 查询可连接的数据集
    for base_table in base_tables:
        print(f"\n查询基表: {base_table}")
        base_table_name = Path(base_table).stem
        base_table_info = get_joinable_datasets(base_table)

        # 保存查询结果
        output_path = Path(f'join_info/{base_table_name}_join_info.csv')
        if output_path.exists():
            old_info = pd.read_csv(output_path)
            base_table_info = pd.concat([old_info, base_table_info], ignore_index=True)
            base_table_info = base_table_info.drop_duplicates(subset=['id']) # 去重
        base_table_info.to_csv(output_path, index=False)
        print(f'join data saved to {output_path}')
    
    # 2. 下载查询到的数据集
    if download:
        for base_table in base_tables:
            print(f"\n下载基表{base_table}的连接数据集：")
            base_table_name = Path(base_table).stem
        
        join_info_path = Path(f'join_info/{base_table_name}_join_info.csv')
        download_join_datasets(base_table_name=base_table_name, join_info_path=join_info_path)

def process_base_table(base_table_name, download=False):

    base_table_info = get_joinable_datasets(f'base_tables/{base_table_name}.csv')

    # 保存查询结果
    output_path = Path(f'join_info/{base_table_name}_join_info.csv')
    if output_path.exists():
        old_info = pd.read_csv(output_path)
        base_table_info = pd.concat([old_info, base_table_info], ignore_index=True)
        base_table_info = base_table_info.drop_duplicates(subset=['id']) # 去重
    base_table_info.to_csv(output_path, index=False)
    print(f'join data saved to {output_path}')

    if download:
        download_join_datasets(base_table_name=base_table_name, join_info_path=output_path)


def get_dataset_description(id:str):
    response = requests.get(
        f'https://auctus.vida-nyu.org/api/v1/metadata/{id}',
    )
    metadata = response.json().get('metadata')
    return metadata.get('description', 'N/A')

def get_base_table(id:str, base_table_name:str, target_attribute:str):
    """ 下载基表，并保存到base_tables目录下，同时保存到base_table_info.csv中 """
    table = download_dataset(id)
    table.to_csv(f'base_tables/{base_table_name}.csv', index=False)
    description = get_dataset_description(id)
    table_info = {
        'base_table_name': base_table_name,
        'target_attribute': target_attribute,
        'description': description
    }

    base_table_info = pd.read_csv('base_table_info.csv')
    new_row = pd.DataFrame([table_info])
    base_table_info = pd.concat([base_table_info, new_row], ignore_index=True)
    base_table_info.to_csv('base_table_info.csv', index=False)


if __name__ == '__main__':
    BASE_TABLE_NAME = 'Poverty'
    id = "datamart.upload.976a384921d34ca1a6304a6c7bc256d7"
    # get_base_table(id, BASE_TABLE_NAME, 'POVALL_2016')
    # 批量下载
    # process_base_tables()

    # 下载单个基表的连接数据集
    # process_base_table(base_table_name=BASE_TABLE_NAME, download=True)
    # download_join_datasets(base_table_name=BASE_TABLE_NAME)
    
    # 下载增强表
    # download_join_datasets(base_table_name='recipe_reviews')
