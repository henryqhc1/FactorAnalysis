"""
2023分享会
author: 邢不行
微信: xbx9585
"""
import pandas as pd

from program.Functions import *
from program.Config import *

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')  # 当前文件的名字

period = 'W53'  # 持仓周期（必填）W53特指周五早盘买入，周三尾盘卖出

# offset = [0]  # offset非必填

select_count = 3  # 选股数量（必填）

flow_fin_cols = ['R_np_atoopc@xbx']  # 流量型财务字段

cross_fin_cols = []  # 截面型财务字段

add_fin_cols = ['R_np_atoopc@xbx_单季']  # 最终需要加到数据上的财务字段


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
    # 获取股票代码
    code = data['股票代码'].iloc[-1]
    # 个股的筹码分布文件路径
    dis_path = chip_dis_path + code + '.csv'
    # 读取筹码分布数据
    dist_df = pd.read_csv(dis_path, encoding='gbk', parse_dates=['交易日期'], skiprows=1)
    dist_df = dist_df.sort_values('交易日期').reset_index(drop=True)
    # 将筹码分布数据与全息数据合并
    data = pd.merge(data, dist_df, how='left', on=['交易日期', '股票代码'])

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
    # ===计算超跌因子：Ret5 & 筹码偏离度
    # 计算Ret5因子
    data['Ret_5'] = data['复权因子'].pct_change(5)
    exg_dict['Ret_5'] = 'last'

    # 计算筹码偏离度因子 = 后复权价格 / 加权平均成本
    data['筹码偏离度'] = data['后复权价格'] / data['加权平均成本'] - 1
    exg_dict['筹码偏离度'] = 'last'

    # ===计算筹码集中度因子
    # 1.计算初始的筹码集中度因子
    # 获取所有分位成本的列名
    cost_col = [c for c in data.columns if '分位成本' in c]
    # 遍历计算每个分位成本相较于当前价格的涨跌幅
    temp_list = []
    for col in cost_col:
        data[col + '_相对涨跌幅'] = data[col] / data['后复权价格'] - 1
        temp_list.append(col + '_相对涨跌幅')
    # 计算筹码分度度因子
    data['筹码集中度'] = data[temp_list].std(axis=1)
    exg_dict['筹码集中度'] = 'last'
    # 2.计算上方 和 下方 的筹码集中度
    data['上方筹码集中度'] = data[data[temp_list] > 0][temp_list].std(axis=1)
    exg_dict['上方筹码集中度'] = 'last'
    data['下方筹码集中度'] = data[data[temp_list] < 0][temp_list].std(axis=1)
    exg_dict['下方筹码集中度'] = 'last'

    # ===计算一些其他需要用到的因子
    # 近一周的跌停状态
    data.loc[data['收盘价'] == data['跌停价'], '异常状态'] = 1
    data['异常状态'].fillna(value=0, inplace=True)
    data['近5日异常状态sum'] = data['异常状态'].rolling(5, min_periods=1).sum()  # 近5个交易日出现异常状态的话，值为1，否则为0
    exg_dict['近5日异常状态sum'] = 'last'

    # 计算价格在最近一年的分位数
    data['价格分位数'] = data['收盘价_复权'].rolling(250).rank(pct=True)
    exg_dict['价格分位数'] = 'last'

    # 市值因子：加点小市值看看效果
    exg_dict['总市值'] = 'last'

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
    # ===剔除情绪面 & 基本面不利的股票
    # 1.剔除情绪面不利的股票
    # 1.1 最近一周不能有跌停
    all_data = all_data[all_data['近5日异常状态sum'] < 1]
    # 1.2 当前价格不能低于最近1年的5分位
    all_data = all_data[all_data['价格分位数'] > 0.05]

    # 2.剔除基本面不利的股票：市值中性化后的归母净利率排名要大于市场一般的股票
    all_data = factor_neutralization(all_data, factor='R_np_atoopc@xbx_单季', neutralize_list=['总市值'])
    all_data['归母净利润_中性化排名'] = all_data.groupby('交易日期')['R_np_atoopc@xbx_单季_中性'].rank(ascending=False,
                                                                                                       method='min',
                                                                                                       pct=True)
    all_data = all_data[all_data['归母净利润_中性化排名'] < 0.5]

    # ===构建策略需要的因子
    # 1.超跌因子：Ret5 & 筹码偏离度 等权组合
    all_data['Ret_5_排名'] = all_data.groupby('交易日期')['Ret_5'].rank(ascending=True)
    all_data['筹码偏离度_排名'] = all_data.groupby('交易日期')['筹码偏离度'].rank(ascending=True)
    # 因子等权组合
    all_data['超跌因子'] = all_data['Ret_5_排名'] + all_data['筹码偏离度_排名']
    all_data['超跌因子_排名'] = all_data.groupby('交易日期')['超跌因子'].rank(ascending=True)

    # 2.筹码集中因子
    all_data['筹码集中度_排名'] = all_data.groupby('交易日期')['上方筹码集中度'].rank(ascending=False)

    # 3.市值因子（不是策略的主要因子，想看市值因子的结果再加上）
    all_data['总市值_排名'] = all_data.groupby('交易日期')['总市值'].rank(ascending=True)

    # === 计算复合因子
    # 原版：超跌因子 + 筹码集中度
    all_data['复合因子'] = all_data['超跌因子_排名'] + all_data['筹码集中度_排名']
    # 带市值版：超跌因子 + 筹码集中度 + 市值因子
    # all_data['复合因子'] = all_data['超跌因子_排名'] + all_data['筹码集中度_排名'] + all_data['总市值_排名']

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
