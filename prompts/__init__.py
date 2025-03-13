
from prompts import llm_prompts


def get_prompt(prompt_type: str, **kwargs) -> str:
    """获取指定类型的提示词模板"""
    templates = {
        'system': llm_prompts.system_prompt,
        'user': llm_prompts.user_prompt,
        'generate_answer_prompt': llm_prompts.generate_answer_prompt,
        'table_description_prompt': llm_prompts.table_description_prompt,
        'table_relevance_analysis_prompt': llm_prompts.table_relevance_analysis_prompt,
        'format_request_prompt': llm_prompts.format_request_prompt,
        'format_request_with_score_prompt': llm_prompts.format_request_with_score_prompt,
        'format_request_with_analysis_prompt': llm_prompts.format_request_with_analysis_prompt
    }
    if prompt_type == 'table_description_prompt':
        return templates.get(prompt_type, '').format(**kwargs)
    elif prompt_type == 'table_relevance_analysis_prompt':
        table_description = templates.get('table_description_prompt', '').format(**kwargs)
        return templates.get(prompt_type, '').format(table_description=table_description)
    elif prompt_type == 'answer_directly':
        table_description = templates.get('table_description_prompt', '').format(**kwargs)
        generate_answer_request = templates.get('generate_answer_prompt', '')
        format_request = templates.get('format_request_prompt', '')
        return generate_answer_request + table_description + format_request
    elif prompt_type == 'answer_with_analysis':
        table_description = templates.get('table_description_prompt', '').format(**kwargs)
        generate_answer_request = templates.get('generate_answer_prompt', '')
        format_request = templates.get('format_request_with_analysis_prompt', '')
        return generate_answer_request + table_description + format_request
    elif prompt_type == 'generate_answer_prompt':
        table_description = templates.get('table_description_prompt', '').format(**kwargs)
        generate_answer_request = templates.get('generate_answer_prompt', '')
        format_request = templates.get('format_request_prompt', '')

        return generate_answer_request + table_description + format_request
    else:
        return templates.get(prompt_type, '')


def parse_table_analysis_response(response: str) -> dict:
    """从LLM的回答中提取表分析结果
    
    Args:
        response: LLM的原始回答文本
    
    Returns:
        dict: 包含分析结果和结论的字典，格式为：
            {
                'analysis': str,  # 表分析内容
                'conclusion': str  # Yes/No 结论
            }
    """
    try:
        # 初始化结果
        result = {
            'analysis': '',
            'conclusion': ''
        }
        
        # 分行处理
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 提取分析内容
            if line.startswith('Table Analysis:'):
                result['analysis'] = line.replace('Table Analysis:', '').strip()
            
            # 提取结论
            elif line.startswith('Conclusion:'):
                conclusion = line.replace('Conclusion:', '').strip()
                # 标准化结论为 Yes/No
                if conclusion.lower() in ['yes', 'no']:
                    result['conclusion'] = conclusion.capitalize()
                else:
                    print(f"未知的结论格式: {conclusion}")
                    result['conclusion'] = 'No'  # 默认为No
        
        # 验证结果完整性
        if not result['analysis']:
            print("未找到表分析内容")
        if not result['conclusion']:
            print("未找到结论")
            result['conclusion'] = 'No'  # 默认为No
            
        return result
        
    except Exception as e:
        print(f"解析LLM回答时出错: {str(e)}")
        return {
            'analysis': '',
            'conclusion': 'No'  # 出错时默认为No
        }

