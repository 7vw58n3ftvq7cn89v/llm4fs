from analyzer import FilterBasedAnalyzer, WrapperBasedAnalyzer, ModelBasedAnalyzer
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加项目根目录到系统路径
from table_process import Table

def test_filter_based_analyzer(base_table, augment_table, target_attribute):
    analyzer = FilterBasedAnalyzer()
    analyzer.analyze_features(base_table, augment_table, target_attribute)

def test_wrapper_based_analyzer(base_table, augment_table, target_attribute):
    analyzer = WrapperBasedAnalyzer()
    analyzer.analyze_features(base_table, augment_table, target_attribute)

def test_model_based_analyzer(base_table, augment_table, target_attribute):
    analyzer = ModelBasedAnalyzer()
    analyzer.analyze_features(base_table, augment_table, target_attribute)


def test_analyzers(analyzer_list:list, base_table:Table, augment_table:Table, target_attribute:str):
    for analyzer in analyzer_list:
        print(f"Analyzing with {analyzer.__class__.__name__}")
        feature_list = analyzer.analyze_features(base_table, augment_table, target_attribute)
        print(f"Analyzed features: {feature_list}")


if __name__ == "__main__":
    
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

    analyzer_list = [FilterBasedAnalyzer(), 
                    #  WrapperBasedAnalyzer(), 
                    #  ModelBasedAnalyzer()
                     ]
    test_analyzers(analyzer_list, base_table, augment_table, TARGET_ATTRIBUTE)