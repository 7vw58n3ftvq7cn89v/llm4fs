from prompts import get_prompt, parse_table_analysis_response

def test_table_description_prompt():
    print(get_prompt('table_description_prompt', **{
        "base_table_description": "base_table_description",
        "augment_table_description": "augment_table_description",
        "task_type": "regression",
        "target_attribute": "target_attribute"
    }))
    print("--------------------------------")
    print(get_prompt('generate_answer_prompt', **{
        "base_table_description": "base_table_description",
        "augment_table_description": "augment_table_description",
        "task_type": "regression",
        "target_attribute": "target_attribute"
    }))
    print("--------------------------------")
    print("table_relevance_analysis_prompt:")
    print(get_prompt('table_relevance_analysis_prompt', **{
        "base_table_description": "base_table_description",
        "augment_table_description": "augment_table_description",
        "task_type": "regression",
        "target_attribute": "target_attribute"
    }))

def test_parse_table_analysis_response():
    response = "Table Analysis: analyzing the relevance between base table and augment table\nConclusion: Yes"
    print(parse_table_analysis_response(response))

if __name__ == "__main__":
    test_table_description_prompt()
    test_parse_table_analysis_response()