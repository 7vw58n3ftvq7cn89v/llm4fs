
### 用大模型指导feature的选择

输入：表格描述、表格的特征集合、每个特征的代表性样本


## 数据收集

base_tables: 
使用[get_base_tables.py](./data/get_base_tables.py)获取；
从kaggle等平台收集

join_info、join_datasets: 
使用[get_datasets.py](./data/get_datasets.py)获取；
调用nyu-auctus平台的api收集可连接表；


## 表格增强

- 表格选择
- 特征选择

### 调用LLM分析特征

[analyzer.py](./feature_analyze/analyzer.py)

## LLM prompts

[prompts.py](./prompts/prompts.py)


system_prompt: 系统提示词，设定任务要求与输出格式
user_prompt: 用户提示词，输入具体的表格的描述

generate_answer_prompt: 设定要求与输出格式

table_description_prompt: 从Table中提取表格信息，[实现的函数](./table_process.py#get_table_description)
- 表格的名称
- 表格的schema信息(列名、数据类型、数据示例)
- 元数据：对表格的描述


## table管理
[table_process.py](./table_process.py)

TODO:
- 优化get_table_info方法，保存已计算的列信息，每次调用，计算未计算的新增列的信息
- BaseTable class: 基表，需要保留df
- AugmentTable class: 增强表，不需要保留df

## 评估策略

1. 对比挑选与未挑选的特征质量

2. 对比其他方法，每种方法维护一个current_df，比较不同方法的性能提升
baseline：
- random
- mutual
- all

评估指标：
- 特征数
- 性能提升
- 速度


## Next Step
- [ ] 增加LLM初筛，得到过滤后的表集合
- [ ] prompt精细化：[prompts](#LLM-prompts)
    - [ ] 表格相关性分析的准确度不够，优化判断的方式
    - [ ] 特征选择：调整prompt，逐个分析特征相关性
- [ ] 数据收集：从可连接表中扩充连接键，aqe-nta
- [ ] 数据收集：寻找新的基表与增强表集合
- [ ] 支持更多筛选方法[analyzer](./feature_analyze/analyzer.py)：
    - [ ] Filter-based:基于统计信息过滤
    - [ ] Wrapper-based：迭代选取或删除
    - [ ] Model-based：基于模型重要性排序

## TODO

大目标：
1. prompt层面：提升LLM筛选特征的准确度
2. 表格筛选：筛选相关的表格用于增强，探索LLM筛选表格的能力
3. 实验数据：高质量的基表+增强表集合，目前只有schools一个数据集，且增强表集合质量不高


### Maybe list

- [ ] 数据收集：从可连接表中扩充连接键
- [ ] 表格描述：包括空值的比例
- [ ] 使用不同大模型对比
- [ ] Table类的设计：BaseTable、AugmentTable [table_process.py](./table_process.py)


## Change log
- 2025-02-26: 增加Analyzer类
- 2025-02-28: 修改processor类，增加method参数，支持不同的评估策略 
- 2025-02-28: 增加prompt管理
- 2025-03-04: 修改processor类，增加method参数，支持不同的评估策略 
- 2025-03-04: processor内置不同的基线方法
- 2025-03-04: 解决schools数据集增强表中的数据泄露问题
- 2025-03-05: 回答解析问题：2025-02-28 01:01:05,240 - __main__ - ERROR - Method llm failed: Failed to parse response: Expecting ',' delimiter: line 7 column 5 (char 1305)
- 2025-03-05: 增加数据集元数据下载
- 2025-03-06: Table类增加join_features方法，优化表格的连接以及重名的处理
- 2025-03-07: table_description_prompt的表格描述生成代码
- 2025-03-08: 处理left join导致行数扩充的问题
- 2025-03-09: 增加analyzer的表格相关性分析部分
- 2025-03-09: 增加processor的表格相关性分析部分
- 2025-03-11: 优化get_table_info方法，保存已计算的列信息，每次调用，计算未计算的新增列的信息
- 2025-03-12: 增加推理模型的支持
