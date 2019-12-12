# coding=UTF-8
from user_pay_get_last_week import user_pay_get_last_week
from user_pay_get_last_two_weeks import user_pay_get_last_two_weeks
from user_pay_get_last_three_weeks import user_pay_get_last_three_weeks
from user_pay_split_by_month import user_pay_split_by_month
from user_pay_split_by_shop import user_pay_split_by_shop


#预处理数据，按照月份星期和商家进行划分
#这里不管用户浏览数据
def preprocess():
    print 'Step1 processing...'
    user_pay_split_by_month()
    print 'Step2 processing...'
    user_pay_get_last_week()
    print 'Step3 processing...'
    user_pay_get_last_two_weeks()
    print 'Step4 processing...'
    user_pay_get_last_three_weeks()
    print 'Step5 processing...'
    user_pay_split_by_shop()
