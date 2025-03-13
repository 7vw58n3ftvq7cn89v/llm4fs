import pandas as pd
from openai import OpenAI
from typing import Dict
from pathlib import Path



PROMPT = """请分析以下表格数据结构，先为每个列生成简洁的技术描述，然后总结整个表格。
数据特征：
{columns_str}

生成要求：
1. 按这个格式描述每个列：
   [列名]: (数据类型) - 列描述
   
2. 根据列的描述，对整个表格进行描述
"""



class CSVDescriber:
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _read_data(self, file_path: str, sample_size: int = 3) -> Dict:
        """读取CSV数据并提取关键信息"""
        df = pd.read_csv(file_path)
        return {
            "columns": df.columns.tolist(),
            "samples": df.head(sample_size).to_dict(orient='records'),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "total_rows": len(df)
        }

    def _generate_prompt(self, data: Dict) -> str:
        """构建大模型提示词"""
        columns_str = "\n".join([
            f"- {col} ({data['dtypes'][col]}): {', '.join(map(str, [rec[col] for rec in data['samples']]))}"
            for col in data['columns']
        ])
        
        return PROMPT.format(columns_str=columns_str)

    def _parse_response(self, response: str) -> Dict:
        """解析大模型响应"""
        result = {"columns": {}, "summary": ""}
        current_section = None
        
        for line in response.split('\n'):
            if ':' in line and '(' in line:
                col_name = line.split(':')[0].strip()
                result["columns"][col_name] = line
            elif line.startswith("数据规模") or line.startswith("主要包含"):
                current_section = "summary"
                result["summary"] += line + "\n"
            elif current_section == "summary":
                result["summary"] += line + "\n"
        
        return result

    def describe(self, file_path: str) -> Dict:
        """主处理流程"""
        # 读取数据
        data_info = self._read_data(file_path)
        
        # 调用大模型
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": self._generate_prompt(data_info)
                }]
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"API调用失败: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    API_KEY: str = "sk-vfvgqjyrqycvufkfhxcrrdwwwhmkvyaktyabieuewslzwggb"
    BASE_URL: str = "https://api.siliconflow.cn/v1"
    MODEL_NAME: str = "deepseek-ai/DeepSeek-V2.5"
    BASE_TABLE = Path("../data/base_tables/Air_Quality.csv")
    describer = CSVDescriber(api_key=API_KEY, base_url=BASE_URL, model=MODEL_NAME)
    
    try:
        result = describer.describe(BASE_TABLE)
        print("\n列描述：")
        for col, desc in result['columns'].items():
            print(f"{desc}\n")
        
        print("\n表格总结：")
        print(result['summary'])
    except Exception as e:
        print(f"处理失败: {str(e)}")