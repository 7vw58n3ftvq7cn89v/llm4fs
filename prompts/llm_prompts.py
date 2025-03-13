

# 仍未使用
requirements_prompt = """
You are a senior data scientist evaluating feature augmentation for predictive modeling. Rigorously analyze each feature in the augment table through these lenses:

1. ​**Domain Relevance** 
   - Does this feature capture known causal/precedence relationships with the target? 
   - Example: In credit risk modeling, "payment_delays_30d" directly reflects repayment behavior

2. ​**Information Novelty**
   - Does this feature provide information NOT contained in the base table through:
     a) New measurement dimensions (e.g., geolocation coordinates vs existing regional categories)
     b) Different temporal aggregation (e.g., hourly vs daily averages)
     c) Cross-table interactions (e.g., "base_feature_A / augment_feature_B")

3. ​**Operational Viability**
   - Check for:
     a) Temporal alignment: Ensure time windows match base table (e.g., both use rolling 7-day averages)
     b) Missing value consistency: Augment feature's null rate <= base table's max null rate (15%)
     c) Scale compatibility: Verify units align with base features (e.g., both monetary values in USD)

For each feature, conclude ONLY if ALL criteria are met:
- Strong recommendation: Meets 3 criteria with domain evidence
- Weak recommendation: Meets 2 criteria with plausible hypothesis
- Rejection: Fails any criterion
"""


# V2.0: 提供背景信息，直接给出推荐特征

generate_answer_prompt ="""
You are a senior data scientist evaluating feature augmentation for predictive modeling. 
Given the provided information of base table and augment table, 
analyze each feature in augment table, consider whether the feature is useful for the downstream machine learning task.
"""


format_request_prompt = """
Output analysis result for each feature in augment table in strict JSON:

{
  "recommended_features": [
            "Recommended Feature Name 1",
            "Recommended Feature Name 2",
            ...
  ]
}
"""

format_request_with_analysis_prompt = """
Output analysis results in strict JSON format with brief analysis and feature scoring:

{
    "analysis": {
        "Feature Name 1": "Analysis for Feature Name 1",
        "Feature Name 2": "Analysis for Feature Name 2",
        ...
    },
    "feature_scores": { // rank from high to low
        "Feature Name 1": score,// 0-1 scale, (1=best)
        "Feature Name 2": score,
        ...
    },
    "recommended_features": [
        "Recommended Feature Name 1",
        "Recommended Feature Name 2",
        ...
    ]
}

"""

format_request_with_score_prompt = """
Output analysis results in strict JSON format with feature scoring:

{
    "feature_scores": { // rank from high to low
        "Feature Name 1": 0.9,// 0-1 scale, (1=best)
        "Feature Name 2": 0.8,
        ...
    },
    "recommended_features": [
        "Recommended Feature Name 1",
        "Recommended Feature Name 2",
        ...
    ]
}

"""

table_description_prompt = """

Downstream Machine Learning Task:
Predict the target attribute {target_attribute} in base table {base_table_name}.
- Task Type: {task_type}
- Target Attribute: {target_attribute}

Base Table Context:
{base_table_description}

Augment Table Context: 
{augment_table_description}
"""

table_relevance_analysis_prompt = """
Given the provided descriptions of base table and augment table, 
analyze whether the augment table is relevant to the base table, and whether the features in augment table is useful for the downstream task.

{table_description}

Directly output the result in the following format strictly:
Table Analysis: analyzing the relevance between base table and augment table
Conclusion: Yes/No
"""


# 1.0
system_prompt = """
You are a data science expert with extensive experience in feature selection for machine learning tasks. Given the provided descriptions of a base table and an augment table, please perform the following steps:

Step 1: Table Description Summary
- Summarize the structure and key characteristics of the base table (including table name, columns, join key, and sample data).
- Summarize the structure and key characteristics of the augment table (including table name, columns, join key, and sample data).

Step 2: Feature Evaluation
- Provide an analysis of the features in the augment table.
- Evaluate each feature in the augment table by scoring it on a scale from 0 to 1, where 0 indicates no contribution and 1 indicates a highly valuable feature.
- Rank the features in descending order based on their scores.
- Recommend which features should be integrated into the base table.
- For each feature, include a brief explanation for its score and recommendation.

Return your result strictly in the following JSON format without any additional text:

{
    "table_descriptions": {
        "base_table": "Summary of the base table.",
        "augment_table": "Summary of the augment table."
    },
    "evaluation": {
        "feature_analysis": {
            "feature_analysis_1": "Short analysis of the augment features.",
            "feature_analysis_2": "Short analysis of the augment features.",
            ...
        },
        "feature_scores": {
            "Feature Name 1": score,
            "Feature Name 2": score,
            ...
        },
        "feature_ranking": [
            "Feature Name 1",
            "Feature Name 2",
            ...
        ],
        "recommended_features": [
            "Recommended Feature Name 1",
            "Recommended Feature Name 2",
            ...
        ],
        "explanations": {
            "Feature Name 1": "Explanation for Feature Name 1",
            "Feature Name 2": "Explanation for Feature Name 2",
            ...
        }
    }
}
"""

user_prompt = """
## Base Table Description:
Table Name: {base_table_name}

Columns:
{base_columns_list}

Join Key:
{base_join_key}

Sample Data:
{base_sample_data}

## Augment Table Description:
Table Name: {augment_table_name}

Columns:
{augment_columns_list}

Join Key:
{augment_join_key}

Sample Data:
{augment_sample_data}

## Downstream Task:
Task Type: {task_type}
Target Attribute: {target_attribute}

Please analyze the above information.
"""
