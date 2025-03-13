from openai import OpenAI
import os
import sys
import pandas as pd
import json
import re
from pathlib import Path
from abc import ABC, abstractmethod
import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.logger import get_logger
from table_process import Table

logger = get_logger(__name__)

RECENT_RESPONSE = """
{
    "feature_scores": {
        "DISTRICT": 0.1,
        "SCHOOL LEVEL*": 0.2,
        "PEER INDEX*": 0.3,
        "2009-2010 OVERALL SCORE": 0.9,
        "2009-2010 ENVIRONMENT CATEGORY SCORE": 0.6,
        "2009-2010 ENVIRONMENT GRADE": 0.5,
        "2009-2010 PERFORMANCE CATEGORY SCORE": 0.7,
        "2009-2010 PERFORMANCE GRADE": 0.6,
        "2009-2010 PROGRESS CATEGORY SCORE": 0.8,
        "2009-2010 PROGRESS GRADE": 0.7,
        "2009-2010 ADDITIONAL CREDIT": 0.4,
        "2008-09 PROGRESS REPORT GRADE": 0.5,
        "2011-2012 OVERALL GRADE": 0.8,
        "2011-2012 OVERALL SCORE": 0.9,
        "2011-12 OVERALL PERCENTILE": 0.7,
        "2011-2012 PROGRESS CATEGORY SCORE": 0.8,
        "2011-2012 PROGRESS GRADE": 0.7,
        "2011-2012 PERFORMANCE CATEGORY SCORE": 0.7,
        "2011-2012 PERFORMANCE GRADE": 0.6,
        "2011-2012 ENVIRONMENT CATEGORY SCORE": 0.6,
        "2011-2012 ENVIRONMENT GRADE": 0.5,
        "2011-2012 COLLEGE AND CAREER READINESS SCORE": 0.4,
        "2011-2012 COLLEGE AND CAREER READINESS GRADE": 0.3,
        "2011-2012 ADDITIONAL CREDIT": 0.4,
        "2010-11 PROGRESS REPORT GRADE": 0.5,
        "2009-10 PROGRESS REPORT GRADE": 0.5
    },
    "feature_ranking": [
        "2009-2010 OVERALL SCORE",
        "2011-2012 OVERALL SCORE",
        "2009-2010 PROGRESS CATEGORY SCORE",
        "2011-2012 PROGRESS CATEGORY SCORE",
        "2011-12 OVERALL PERCENTILE",
        "2009-2010 PERFORMANCE CATEGORY SCORE",
        "2011-2012 PERFORMANCE CATEGORY SCORE",
        "2009-2010 PROGRESS GRADE",
        "2011-2012 PROGRESS GRADE",
        "2009-2010 PERFORMANCE GRADE",
        "2011-2012 PERFORMANCE GRADE",
        "2009-2010 ENVIRONMENT CATEGORY SCORE",
        "2011-2012 ENVIRONMENT CATEGORY SCORE",
        "2009-2010 ENVIRONMENT GRADE",
        "2011-2012 ENVIRONMENT GRADE",
        "2009-2010 ADDITIONAL CREDIT",
        "2011-2012 ADDITIONAL CREDIT",
        "2008-09 PROGRESS REPORT GRADE",
        "2010-11 PROGRESS REPORT GRADE",
        "2009-10 PROGRESS REPORT GRADE",
        "PEER INDEX*",
        "SCHOOL LEVEL*",
        "DISTRICT",
        "2011-2012 COLLEGE AND CAREER READINESS SCORE",
        "2011-2012 COLLEGE AND CAREER READINESS GRADE"
    ],
    "recommended_features": [
        "2009-2010 OVERALL SCORE",
        "2011-2012 OVERALL SCORE",
        "2009-2010 PROGRESS CATEGORY SCORE",
        "2011-2012 PROGRESS CATEGORY SCORE",
        "2011-12 OVERALL PERCENTILE",
        "2009-2010 PERFORMANCE CATEGORY SCORE",
        "2011-2012 PERFORMANCE CATEGORY SCORE",
        "2009-2010 PROGRESS GRADE",
        "2011-2012 PROGRESS GRADE",
        "2009-2010 PERFORMANCE GRADE",
        "2011-2012 PERFORMANCE GRADE",
        "2009-2010 ENVIRONMENT CATEGORY SCORE",
        "2011-2012 ENVIRONMENT CATEGORY SCORE",
        "2009-2010 ENVIRONMENT GRADE",
        "2011-2012 ENVIRONMENT GRADE"
    ]
}
"""

class BaseLLMClient(ABC):
    """LLM客户端的基类"""
    @abstractmethod
    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        """获取LLM响应的抽象方法"""
        pass

class OpenAIClient(BaseLLMClient):
    """OpenAI API客户端"""
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

class OllamaClient(BaseLLMClient):
    """Ollama本地模型客户端"""
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "deepseek-r1:8b"):
        self.base_url = base_url
        self.model_name = model_name
        
    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        
        # 组合system prompt和user prompt
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
            raise Exception(f"Ollama API调用失败: {str(e)}")

class LLMFeatureAnalyzer:
    def __init__(self, llm_client: BaseLLMClient):
        """
        初始化特征分析器
        Args:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client
        self.metrics = {
            "classification": ["accuracy", "f1", "precision", "recall"],
            "regression": ["rmse", "mae", "r2"]
        }
        
    def analyze_features(self, 
                         base_df: pd.DataFrame, 
                         base_name: str,
                         augment_table: Table,
                         target_attribute: str
                    ):
        """
        分析特征重要性的主入口方法
        """
        logger.info("start analyzing features with LLM")

        # 提取表格信息
        base_table_info = self._extract_table_info(base_df, base_name, augment_table.left_key)
        augment_table_info = self._extract_table_info(augment_table.get_df(), augment_table.name, augment_table.right_key)
        task_info = self._infer_task_info(base_df, target_attribute)
        
        # 生成并获取LLM响应
        prompt = self._generate_prompt(augment_table_info, base_table_info, task_info)
        response = self._get_llm_response(prompt) 
        # logger.info(f"prompt: {prompt}, LLM response: {response}")
        # response = RECENT_RESPONSE #节省token，暂时直接返回先前的response
        
        # 解析响应
        logger.info("Successfully get LLM response")
        return self._parse_llm_response(response)
    
    def _extract_table_info(self, augment_df: pd.DataFrame, table_name, join_key: str = None):
        """提取表格信息的内部方法"""
        return {
            "description": f"{table_name}包含的字段及其示例数据。",
            "columns": augment_df.columns.tolist(),
            "sample_data": augment_df.head(3).to_dict(orient="records"),
            "table_name": table_name,
            "join_key": join_key
        }
    
    def _infer_task_info(self, df, target_attribute):
        """推断任务类型的内部方法"""
        if target_attribute not in df.columns:
            raise ValueError(f"目标属性 {target_attribute} 不在数据表中。")

        if df[target_attribute].dtype == 'object' or df[target_attribute].nunique() <= 10:
            task_type = "分类"
        else:
            task_type = "回归"

        return {"task_type": task_type, "target_attribute": target_attribute}
    
    def _generate_prompt(self, augment_table_info, base_table_info, task_info) -> str:
        """生成提示词的内部方法"""

        base_table_section = """
        ## 基表描述：
        表格名称：{base_table_name}

        列名：
        {base_columns_list}

        连接键：
        {base_join_key}

        示例数据：
        {base_sample_data}
        """

        enhanced_table_section = """
        ## 增强表描述：
        表格名称：{enhanced_table_name}

        列名：
        {enhanced_columns_list}

        连接键：
        {enhanced_join_key}

        示例数据：
        {enhanced_sample_data}
        """

        task_section = """
        ## 下游任务描述：
        任务类型：{task_type}
        目标属性：{target_attribute}
        """

        # requirement_section = """
        # ## 要求
        # 请对增强表中的特征进行评分，并输出以下内容，要求回答格式为 JSON：
        # {{
        #     "feature_scores": {{
        #         "特征名1": 评分,
        #         "特征名2": 评分,
        #         ...
        #     }},
        #     "feature_ranking": [
        #         "特征名1",
        #         "特征名2",
        #         ...
        #     ],
        #     "recommended_features": [
        #         "推荐特征名1",
        #         "推荐特征名2",
        #         ...
        #     ]
        # }}
        # """

        # 提取基表信息
        base_table_name = base_table_info.get("table_name", "未知表格")
        base_columns = base_table_info.get("columns", [])
        base_sample_data = base_table_info.get("sample_data", [])
        base_columns_list = "\n".join([f"- {col}" for col in base_columns])
        base_join_key = base_table_info.get("join_key", "未知连接键")
        base_sample_data_str = "\n".join([
            f"  {i+1}. {row}" for i, row in enumerate(base_sample_data)
        ])

        # 提取增强表信息
        enhanced_table_name = augment_table_info.get("table_name", "未知表格")
        enhanced_columns = augment_table_info.get("columns", [])
        enhanced_sample_data = augment_table_info.get("sample_data", [])
        enhanced_columns_list = "\n".join([f"- {col}" for col in enhanced_columns])
        enhanced_join_key = augment_table_info.get("join_key", "未知连接键")
        enhanced_sample_data_str = "\n".join([
            f"  {i+1}. {row}" for i, row in enumerate(enhanced_sample_data)
        ])

        # 提取任务信息
        task_type = task_info.get("task_type", "未知任务类型")
        target_attribute = task_info.get("target_attribute", "未知目标属性")

        # 格式化各个部分
        base_table_section = base_table_section.format(
            base_table_name=base_table_name,
            base_columns_list=base_columns_list,
            base_join_key=base_join_key,
            base_sample_data=base_sample_data_str
        )

        enhanced_table_section = enhanced_table_section.format(
            enhanced_table_name=enhanced_table_name,
            enhanced_columns_list=enhanced_columns_list,
            enhanced_join_key=enhanced_join_key,
            enhanced_sample_data=enhanced_sample_data_str
        )

        task_section = task_section.format(
            task_type=task_type,
            target_attribute=target_attribute
        )

        # 合并所有部分
        prompt = "\n".join([
            # intro_section,
            base_table_section,
            enhanced_table_section,
            task_section
            # requirement_section
        ])

        return prompt

        
    def _get_llm_response(self, user_prompt):
        """获取LLM响应的内部方法"""
        system_prompt = """
        你是一名数据科学专家。以下是一个基表和一个增强表的描述，这些表格将用于下游的机器学习任务。
        请分析增强表中的哪些特征集成到基表后对下游任务可能有积极影响，评估增强表中每个特征对下游任务的贡献，
        并返回以下内容：增强表中的每个特征的评分估计、特征排名以及推荐选择的特征，返回格式为json。
        注意从总体考虑表格之间的关系和特征之间的关联性，两个表格之间可能毫不相关。
        
        严格按照以下JSON格式返回，不要添加任何其他文本:
        {
            "feature_scores": {
                "特征名1": 评分,
                "特征名2": 评分,
                ...
            },
            "feature_ranking": [
                "特征名1",
                "特征名2",
                ...
            ],
            "recommended_features": [
                "推荐特征名1",
                "推荐特征名2",
                ...
            ]
        }
        """
        return self.llm_client.get_completion(system_prompt, user_prompt)
    
    def _parse_llm_response(self, response):
        """解析LLM响应的内部方法"""
        # print(f"response: {response}")
        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            json_str = match.group(1) if match else response.strip()

            response_data = json.loads(json_str)
            
            # response_data = json.loads(response)

            return {
                "feature_scores": response_data.get("feature_scores", {}),
                "feature_ranking": response_data.get("feature_ranking", []),
                "recommended_features": response_data.get("recommended_features", [])
            }
        except Exception as e:
            raise ValueError(f"解析响应失败: {e}")

def main():
    # 配置参数
    USE_OLLAMA = False  # 切换是否使用Ollama
    
    if USE_OLLAMA:
        # 使用Ollama本地模型
        llm_client = OllamaClient(
            base_url="http://localhost:11434",
            model_name="llama3"  # 或其他已下载的模型
            # model_name="deepseek-r1:8b"  # 或其他已下载的模型
        )
    else:
        # 使用OpenAI API
        llm_client = OpenAIClient(
            api_key="sk-tuhtuuoysbjmhqnmvuonlpbndfauxgisopueoexiduzvdplp",
            # api_key="sk-vfvgqjyrqycvufkfhxcrrdwwwhmkvyaktyabieuewslzwggb",
            base_url="https://api.siliconflow.cn/v1",
            model_name="deepseek-ai/DeepSeek-V2.5"
        )
    
    BASE_TABLE = "schools"
    AUGMENT_TABLE_NAME = "Citywide_Progress"
    TARGET_ATTRIBUTE = "2009-2010 OVERALL GRADE"
    
    # 读取数据
    base_df = pd.read_csv(Path(f"data/{BASE_TABLE}.csv"))
    augment_table = Table(
        data_path=Path(f"data/{AUGMENT_TABLE_NAME}.csv"),
        score=1,
        dataset_name=AUGMENT_TABLE_NAME,
        left_column_name='DBN',
        right_column_name='DBN',
        size=0
    )
    
    # 创建分析器实例
    analyzer = LLMFeatureAnalyzer(llm_client=llm_client)
    
    try:
        # 执行分析
        results = analyzer.analyze_features(
            base_df=base_df,
            base_name=BASE_TABLE,
            augment_table=augment_table,
            target_attribute=TARGET_ATTRIBUTE
        )
        
        # 输出结果
        print("推荐特征：")
        print(results["recommended_features"])
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")

if __name__ == "__main__":
    main()



