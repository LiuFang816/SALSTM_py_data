# coding=UTF-8
from preprocess.data_process import preprocess
from rule.submission_by_rule import get_result_by_last_week_with_weight
from rule.extra_fix import extra_fix

#预处理数据，获取切分后的数据
preprocess()
#算结果
get_result_by_last_week_with_weight()
#最后做少量人工调整
extra_fix()
