from openai import OpenAI
import os
import sys
import pandas as pd
import json
import re
from pathlib import Path
from abc import ABC, abstractmethod
import requests
import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infrastructure.logger import get_logger
from table_process import Table
from prompts import get_prompt, parse_table_analysis_response
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

class BaseLLMClient(ABC):
    """Base class for LLM client"""
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    @abstractmethod
    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Abstract method for obtaining LLM response"""
        pass

class OpenAIClient(BaseLLMClient):
    """OpenAI API Client"""
    def __init__(self, api_key: str, base_url: str, model_name: str):
        super().__init__(api_key, base_url, model_name)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_completion(self, system_prompt: str="", user_prompt: str="", inference: bool=False) -> str:
        stream_mode = inference
        prompt = f"{system_prompt}\n\n{user_prompt}"
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=stream_mode
        )
        if not inference:
            return completion.choices[0].message.content
        
        # 推理模型，提取推理内容和回答内容
        reasoning_content = ""
        content = ""
        
        for chunk in completion:
            # If chunk.choices is empty, print usage
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                else:
                    content += delta.content
        
        return content


class OllamaClient(BaseLLMClient):
    """Ollama Local Model Client"""
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "deepseek-r1:8b"):
        super().__init__(base_url, model_name)
        
    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        
        # Combine system prompt and user prompt
        combined_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
        
        payload = {
            "model": self.model_name,
            "prompt": combined_prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API call failed: {str(e)}")

class Analyzer(ABC):
    """Base class for feature analyzers"""
    
    @abstractmethod
    def analyze_features(self, 
                        base_table: Table,
                        augment_table: Table,
                        target_attribute: str):
        """
        Main method for analyzing feature importance.
        
        Args:
            base_df: Base dataframe
            base_name: Name of the base table
            augment_table: Table object containing augmentation data
            target_attribute: Target column name for prediction
            
        Returns:
            Dict containing analysis results
        """
        pass

class FilterBasedAnalyzer(Analyzer):
    def __init__(self):
        """初始化过滤式特征分析器"""
        self.correlation_threshold = 0.1  # 相关性阈值
        self.max_missing_ratio = 0.3     # 最大缺失值比例

    def analyze_features(self, 
                        base_table: Table,
                        augment_table: Table,
                        target_attribute: str):
        """
        使用过滤方法分析特征重要性
        
        Args:
            base_table: 基础表对象
            augment_table: 增强表对象
            target_attribute: 目标变量名称
            
        Returns:
            Dict: 包含推荐特征列表的字典
        """
        logger.info("start filtering features")
        
        # 获取增强表的所有特征
        augment_features = augment_table.get_columns()
        if augment_table.right_key in augment_features:
            augment_features.remove(augment_table.right_key)
            
        # 连接数据
        merged_df = augment_table.join_features(base_table.get_df(), augment_features)
        
        recommended_features = []
        target_series = merged_df[target_attribute]
        
        for feature in augment_features:
            # 跳过目标变量
            if feature == target_attribute or feature == augment_table.right_key:
                continue
                
            feature_series = merged_df[feature]
            
            # 检查缺失值比例
            missing_ratio = feature_series.isnull().mean()
            if missing_ratio > self.max_missing_ratio:
                continue
                
            # 计算相关性
            correlation = self._calculate_correlation(feature_series, target_series)
            
            if abs(correlation) >= self.correlation_threshold:
                recommended_features.append(feature)
                
        logger.info(f"selected {len(recommended_features)} features")
        
        return {
            "recommended_features": recommended_features
        }
    
    def _calculate_correlation(self, feature_series:pd.Series, target_series:pd.Series):
        """计算特征与目标变量的相关性
        
        Args:
            feature_series: 特征列
            target_series: 目标变量列
            
        Returns:
            float: 相关系数
        """
        try:
            # 对于数值型特征，使用皮尔逊相关系数
            if pd.api.types.is_numeric_dtype(feature_series):
                return feature_series.corr(target_series)
            
            # 对于类别型特征，使用克拉默V系数
            else:
                return self._cramers_v(feature_series, target_series)
                
        except Exception as e:
            logger.warning(f"计算相关性时出错: {str(e)}")
            return 0
    
    def _cramers_v(self, x:pd.Series, y:pd.Series):
        """计算克拉默V系数
        
        Args:
            x: 第一个类别变量
            y: 第二个类别变量
            
        Returns:
            float: 克拉默V系数
        """
        try:
            # 将数据转换为类别型
            x = x.astype('category')
            y = y.astype('category')
            
            # 处理空值
            mask = ~(x.isna() | y.isna())
            x = x[mask]
            y = y[mask]
            
            if len(x) == 0 or len(y) == 0:
                return 0
                
            confusion_matrix = pd.crosstab(x, y)
            
            # 检查是否有足够的唯一值
            if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
                return 0
                
            chi2 = stats.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            min_dim = min(confusion_matrix.shape) - 1
            
            if n == 0 or min_dim == 0:
                return 0
                
            return np.sqrt(chi2 / (n * min_dim))
            
        except Exception as e:
            logger.warning(f"计算克拉默V系数时出错: {str(e)}")
            return 0

class WrapperBasedAnalyzer(Analyzer):
    def __init__(self):
        pass

    def analyze_features(self, 
                        base_table: Table,
                        augment_table: Table,
                        target_attribute: str):
        pass

class ModelBasedAnalyzer(Analyzer):
    def __init__(self):
        pass

    def analyze_features(self, 
                        base_table: Table,
                        augment_table: Table,
                        target_attribute: str):
        pass




class LLMFeatureAnalyzer(Analyzer):
    def __init__(self, 
                 api_key: str,
                 base_url: str,
                 model_name: str
            ):
        """
        Initialize the LLM-based feature analyzer.
        Args:
            llm_client: LLM client instance.
        """
        self.llm_client = OpenAIClient(api_key, base_url, model_name)
        self.metrics = {
            "classification": ["accuracy", "f1", "precision", "recall"],
            "regression": ["rmse", "mae", "r2"]
        }
        # self.base_table_description = None # 基表描述

    def analyze_features_old(self, 
                        base_df: pd.DataFrame, 
                        base_name: str,
                        augment_table: Table,
                        target_attribute: str
                    ):
        """
        Implementation of feature analysis using LLM.
        """
        logger.info("start analyzing features with LLM")

        # Extract table information
        base_table_info = self._extract_table_info(base_df, base_name, augment_table.left_key)
        augment_table_info = self._extract_table_info(augment_table.get_df(), augment_table.name, augment_table.right_key)
        task_info = self._infer_task_info(base_df, target_attribute)
        
        # Generate and obtain LLM response
        prompt = self._generate_prompt(augment_table_info, base_table_info, task_info)
        response = self._get_llm_response(prompt) 
        
        # Parse the response
        logger.info("Successfully got LLM response")
        return self._parse_json_response(response)
    
    def get_description_args(self, 
                        base_table: Table,
                        augment_table: Table,
                        target_attribute: str
                    ):
        base_table_info = base_table.get_table_info(schema=True, description=False)
        augment_table_info = augment_table.get_table_info(schema=True, description=False)
        
        task_info = self._infer_task_info(base_table.get_df(), target_attribute)

        description_args = {
            "base_table_description": base_table_info,
            "augment_table_description": augment_table_info,
            "task_type": task_info.get("task_type", "Unknown Task Type"),
            "target_attribute": target_attribute,
            "base_table_name": base_table.name
        }

        return description_args

    def analyze_features(self, 
                        base_table: Table,
                        augment_table: Table,
                        target_attribute: str,
                        table_analysis: bool=False
                    ):
        """
        Implementation of feature analysis using LLM.
        """
        logger.debug("Start analyzing features with LLM")

        if table_analysis:
            table_analysis = self.table_relevance_analysis(base_table, augment_table, target_attribute)
            if table_analysis['conclusion'] == 'No':
                logger.debug(f"LLM suggests that table {augment_table.name} is not relevant to the base table")
                logger.debug(f"LLM analysis: {table_analysis['analysis']}")
                return {'recommended_features': []}

        description_args = self.get_description_args(base_table, augment_table, target_attribute)
        # Generate and obtain LLM response
        prompt = get_prompt('generate_answer_prompt', **description_args)
        response = self._get_llm_response(prompt)   
        
        # Parse the response
        logger.debug("Successfully got LLM response")
        return self._parse_json_response(response)
    
    def table_relevance_analysis(self, 
                                 base_table: Table, 
                                 augment_table: Table, 
                                 target_attribute: str
                            ):
        """分析增强表与基表的相关性"""
        description_args = self.get_description_args(base_table, augment_table, target_attribute)
        prompt = get_prompt('table_relevance_analysis_prompt', **description_args)
        response = self._get_llm_response(prompt)
        return parse_table_analysis_response(response)

    def _extract_table_info(self, df: pd.DataFrame, table_name, join_key: str = None):
        """Internal method to extract table information"""
        return {
            "description": f"{table_name} includes fields and sample data.",
            "columns": df.columns.tolist(),
            "sample_data": df.head(3).to_dict(orient="records"),
            "table_name": table_name,
            "join_key": join_key
        }
    
    def _infer_task_info(self, df: pd.DataFrame, target_attribute: str):
        """Internal method to infer task type"""
        if target_attribute not in df.columns:
            raise ValueError(f"Target attribute {target_attribute} is not in the dataframe.")

        if df[target_attribute].dtype == 'object' or df[target_attribute].nunique() <= 10:
            task_type = "classification"
        else:
            task_type = "regression"

        return {"task_type": task_type, "target_attribute": target_attribute}

 
    def _generate_prompt(self, augment_table_info, base_table_info, task_info) -> str:
        """生成提示词"""
        # 获取用户提示词模板
        prompt_template = get_prompt('user')
        
        # 格式化提示词
        prompt = prompt_template.format(
            base_table_name=base_table_info.get("table_name", "Unknown Table"),
            base_columns_list="\n".join([f"- {col}" for col in base_table_info.get("columns", [])]),
            base_join_key=base_table_info.get("join_key", "Unknown Join Key"),
            base_sample_data="\n".join([f"  {i+1}. {row}" for i, row in enumerate(base_table_info.get("sample_data", []))]),
            
            augment_table_name=augment_table_info.get("table_name", "Unknown Table"),
            augment_columns_list="\n".join([f"- {col}" for col in augment_table_info.get("columns", [])]),
            augment_join_key=augment_table_info.get("join_key", "Unknown Join Key"),
            augment_sample_data="\n".join([f"  {i+1}. {row}" for i, row in enumerate(augment_table_info.get("sample_data", []))]),
            
            task_type=task_info.get("task_type", "Unknown Task Type"),
            target_attribute=task_info.get("target_attribute", "Unknown Target Attribute")
        )
        
        
        return prompt


    def _get_llm_response(self, user_prompt: str):

        # system_prompt = get_prompt_templates('system')

        return self.llm_client.get_completion(user_prompt=user_prompt)

    
    def _parse_json_response(self, response):
        """Internal method to parse LLM response"""
        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            json_str = match.group(1) if match else response.strip()

            response_data = json.loads(json_str, strict=False)

            return response_data
        except Exception as e:
            print(f"Failed response: {response}")
            raise ValueError(f"Failed to parse response: {e}")

def main():
    # Configuration parameters
    # USE_OLLAMA = False  # Toggle whether to use Ollama
    
    # if USE_OLLAMA:
    #     # Using Ollama local model
    #     llm_client = OllamaClient(
    #         base_url="http://localhost:11434",
    #         model_name="llama3"  # or another downloaded model
    #         # model_name="deepseek-r1:8b" 
    #     )
    # else:
    #     # Using OpenAI API
    #     llm_client = OpenAIClient(
    #         # api_key="sk-vfvgqjyrqycvufkfhxcrrdwwwhmkvyaktyabieuewslzwggb",
    #         api_key="sk-tuhtuuoysbjmhqnmvuonlpbndfauxgisopueoexiduzvdplp",
    #         base_url="https://api.siliconflow.cn/v1",
    #         model_name="deepseek-ai/DeepSeek-V2.5"
    #     )
    
    BASE_TABLE = "schools"
    BASE_TABLE_PATH = Path(f"../data/base_tables/{BASE_TABLE}.csv")
    AUGMENT_TABLE_NAME = "2008_-_2009_School_Progress_Report"
    AUGMENT_TABLE_PATH = Path(f"../data/join_datasets/{BASE_TABLE}/{AUGMENT_TABLE_NAME}.csv")
    TARGET_ATTRIBUTE = "2009-2010 OVERALL GRADE"
    
    # Read data
    # base_df = pd.read_csv(BASE_TABLE_PATH)
    augment_table = Table(
        data_path=AUGMENT_TABLE_PATH,
        score=1,
        dataset_name=AUGMENT_TABLE_NAME,
        left_key='DBN',
        right_key='DBN',
        size=0,
        description=BASE_TABLE
    )
    base_table = Table(
        data_path=BASE_TABLE_PATH,
        dataset_name=BASE_TABLE,
        description=BASE_TABLE
    )
    
    # Create analyzer instance
    analyzer = LLMFeatureAnalyzer(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_API_BASE"),
        model_name='qwq-32b'
    )
    
    try:
        print("Table Relevance Analysis:")
        print(analyzer.table_relevance_analysis(
            base_table=base_table,
            augment_table=augment_table,
            target_attribute=TARGET_ATTRIBUTE
        ))

        print("--------------------------------")

        print("Feature Analysis:")
        # Execute analysis
        results = analyzer.analyze_features(
            base_table=base_table,
            augment_table=augment_table,
            target_attribute=TARGET_ATTRIBUTE
        )
        
        # Output results
        print("Analysis Results:")
        print(results)

        
    except Exception as e:
        print(f"Error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
