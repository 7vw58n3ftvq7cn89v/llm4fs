from prompts import get_prompt

def test_table_description_prompt():
    print(get_prompt('table_description_prompt', **{
        "base_table_description": "base_table_description",
        "augment_table_description": "augment_table_description",
        "task_type": "regression",
        "target_attribute": "target_attribute"
    }))
    print(get_prompt('generate_answer_prompt'))

    print("table_relevance_analysis_prompt:")
    print(get_prompt('table_relevance_analysis_prompt', **{
        "base_table_description": "base_table_description",
        "augment_table_description": "augment_table_description",
        "task_type": "regression",
        "target_attribute": "target_attribute"
    }))

if __name__ == "__main__":
    test_table_description_prompt()