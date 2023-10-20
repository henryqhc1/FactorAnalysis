"""
2023分享会
author: 邢不行
微信: xbx9585
"""
from program.Functions import *
from program.Config import *

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')  # 当前文件的名字

period = 3  # 持仓周期（必填）

offset = auto_offset(period)

select_count = 3  # 选股数量（必填）

flow_fin_cols = []  # 流量型财务字段

cross_fin_cols = []  # 截面型财务字段

add_fin_cols = []  # 最终需要加到数据上的财务字段


def special_data():
    """
    处理策略需要的专属数据，非必要。
    :return:
    """

    return


def before_merge_index(data, exg_dict, fill_0_list):
    """
    合并指数数据之前的处理流程，非必要。
    :param data: 传入的数据
    :param exg_dict: resample规则
    :param fill_0_list: 合并指数时需要填充为0的数据
    :return:
    """

    return data, exg_dict, fill_0_list


def merge_single_stock_file(data, exg_dict):
    """
    合并策略需要的单个的数据，非必要。
    :param data:传入的数据
    :param exg_dict:resample规则
    :return:
    """

    return data, exg_dict


def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    """
    合并数据后计算策略需要的因子，非必要
    :param data:传入的数据
    :param fin_data:财报数据（去除废弃研报)
    :param fin_raw_data:财报数据（未去除废弃研报）
    :param exg_dict:resample规则
    :return:
    """

    # Price_Vol_250：最近250日价格标准差

    data['normalized_prices'] = stock_data['收盘价_复权'] / stock_data['收盘价_复权'].iloc[0]
    data['Price_Vol_250'] = data['normalized_prices'].rolling(window=N).std()

    exg_dict['Price_Vol_250'] = 'last'

    return data, exg_dict


def after_resample(data):
    """
    数据降采样之后的处理流程，非必要
    :param data: 传入的数据
    :return:
    """
    return data


def filter_stock(all_data):
    """
    过滤函数，在选股前过滤，必要
    :param all_data: 截面数据
    :return:
    """
    all_data = all_data[all_data['交易日期'] >= '20070101']
    all_data = all_data[
        ~((all_data['股票代码'] == 'sz300156') & (all_data['交易日期'] >= pd.to_datetime('2020-04-10')))]
    all_data = all_data[
        ~((all_data['股票代码'] == 'sz300362') & (all_data['交易日期'] >= pd.to_datetime('2020-04-10')))]
    # =删除不能交易的周期数
    # 删除月末为st状态的周期数
    all_data = all_data[all_data['股票名称'].str.contains('ST') == False]
    # 删除月末为s状态的周期数
    all_data = all_data[all_data['股票名称'].str.contains('S') == False]
    # 删除月末有退市风险的周期数
    all_data = all_data[all_data['股票名称'].str.contains('\*') == False]
    all_data = all_data[all_data['股票名称'].str.contains('退') == False]
    # 删除交易天数过少的周期数
    all_data = all_data[all_data['交易天数'] / all_data['市场交易天数'] >= 0.8]

    all_data = all_data[all_data['下日_是否交易'] == 1]
    all_data = all_data[all_data['下日_开盘涨停'] == False]
    all_data = all_data[all_data['下日_是否ST'] == False]
    all_data = all_data[all_data['下日_是否退市'] == False]
    all_data = all_data[all_data['上市至今交易天数'] > 250]

    return all_data


def select_stock(all_data, count):
    """
    选股函数，必要
    :param all_data: 截面数据
    :param count: 选股数量
    :return:
    """

    all_data['Price_Vol_250_排名'] = all_data.groupby('交易日期')['Price_Vol_250'].rank(ascending=True)

    # === 计算复合因子
    all_data['复合因子'] = all_data['Price_Vol_250_排名']

    # 删除因子为空的数据
    all_data.dropna(subset=['复合因子'], inplace=True)
    # 回测从09年开始
    all_data = all_data[all_data['交易日期'] >= pd.to_datetime('2009-01-01')]

    # 拷贝一份数据用作稳健性测试
    df_for_group = all_data.copy()
    all_data['复合因子_排名'] = all_data.groupby('交易日期')['复合因子'].rank(ascending=True)
    # 按照固定的数量选股
    all_data = all_data[all_data['复合因子_排名'] <= count]
    all_data['选股排名'] = all_data['复合因子_排名']

    return all_data, df_for_group
