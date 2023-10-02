"""
2023分享会
author: 邢不行
微信: xbx9585
选股策略框架
"""
import os

# ===选股参数设定
date_start = '2016-01-01'
date_end = None  # 日期，例如'2022-11-01'，为None时，代表使用到最新的数据
c_rate = 1.2 / 10000  # 手续费
t_rate = 1 / 1000  # 印花税
buy_method = '开盘'  # 设定买入股票的方法，即在什么时候买入。可以是：开盘、均价、09:35收盘价、09:45收盘价、09:55收盘价。其中均价代表当天的均价

backtest_mode = True  # True为回测模式，会把所有的offset都跑完，False为实盘模式，只跑当日应该跑的offset

# 用于增加1_选股数据整理_并行版.py的效率
subset_num = None  # 可以是None或正整数
# !!!!除非对所有代码都非常熟悉，否则强烈建议就用None模式，慢一点但至少是对的!!!!
# 正整数为当前index的最后交易日往回追溯多少个交易日计算因子（然后拼回），可适当提高数据整理速度
# 设置多少需要看策略rolling最大值 / 0.7，很重要！！！
# None为所有数据全算，算力高可无脑全算，！！！当存在ewm计算时，必须开启None模式！！！
# 当有新策略加入时，一定要跑一遍None!!!因为部分因子是第一次计算的。


# 调用那个策略
strategy_file = '筹码因子选股策略3天'

# multiple_process的进程数，由于joblib在windows系统下最多只能利用64线程，所以做了控制
n_job = min(os.cpu_count() - 1, 60)

# ===获取项目根目录
_ = os.path.abspath(os.path.dirname(__file__))  # 返回当前文件路径
root_path = os.path.abspath(os.path.join(_, '..'))  # 返回根目录文件夹

# 周期和offset预运算数据位置
period_offset_file = root_path + r'/data/period_offset.csv'
# 股票日线数据
stock_data_path = r'D:/stock_data/stock-trading-data-pro/'
# 财务数据
fin_path = r'D:/stock_data/stock-fin-data-xbx/'
# 指数数据路径
index_path = r'D:/stock_data/index/sh000300.csv'

# 筹码分布的路径，下载地址：https://www.quantclass.cn/data/stock/stock-chip-distribution
chip_dis_path = r'D:/stock_data/stock-chip-distribution/'
