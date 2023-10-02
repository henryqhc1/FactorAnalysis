"""
2023分享会
author: 邢不行
微信: xbx9585
选股策略框架
"""
import itertools
import os
import shutil
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import re

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
from sklearn.linear_model import LinearRegression


def cal_fuquan_price(df, fuquan_type='前复权', method=None):
    """
    用于计算复权价格
    :param df: 必须包含的字段：收盘价，前收盘价，开盘价，最高价，最低价
    :param fuquan_type: ‘前复权’或者‘后复权’
    :return: 最终输出的df中，新增字段：收盘价_复权，开盘价_复权，最高价_复权，最低价_复权
    """

    # 计算复权因子
    df['复权因子'] = (df['收盘价'] / df['前收盘价']).cumprod()

    # 计算前复权、后复权收盘价
    if fuquan_type == '后复权':
        df['收盘价_复权'] = df['复权因子'] * (df.iloc[0]['收盘价'] / df.iloc[0]['复权因子'])
    elif fuquan_type == '前复权':
        df['收盘价_复权'] = df['复权因子'] * (df.iloc[-1]['收盘价'] / df.iloc[-1]['复权因子'])
    else:
        raise ValueError('计算复权价时，出现未知的复权类型：%s' % fuquan_type)

    # 计算复权
    df['开盘价_复权'] = df['开盘价'] / df['收盘价'] * df['收盘价_复权']
    df['最高价_复权'] = df['最高价'] / df['收盘价'] * df['收盘价_复权']
    df['最低价_复权'] = df['最低价'] / df['收盘价'] * df['收盘价_复权']
    if method and method != '开盘':
        df[f'{method}_复权'] = df[f'{method}'] / df['收盘价'] * df['收盘价_复权']
    # del df['复权因子']
    return df


def get_file_in_folder(path, file_type, contains=None, filters=[], drop_type=False):
    """
    获取指定文件夹下的文件
    :param path: 文件夹路径
    :param file_type: 文件类型
    :param contains: 需要包含的字符串，默认不含
    :param filters: 字符串中需要过滤掉的内容
    :param drop_type: 是否要保存文件类型
    :return:
    """
    file_list = os.listdir(path)
    file_list = [file for file in file_list if file_type in file]
    if contains:
        file_list = [file for file in file_list if contains in file]
    for con in filters:
        file_list = [file for file in file_list if con not in file]
    if drop_type:
        file_list = [file[:file.rfind('.')] for file in file_list]

    return file_list


# 导入指数
def import_index_data(path, back_trader_start=None, back_trader_end=None):
    """
    从指定位置读入指数数据。指数数据来自于：program_back/构建自己的股票数据库/案例_获取股票最近日K线数据.py
    :param back_trader_end: 回测结束时间
    :param back_trader_start: 回测开始时间
    :param path:
    :return:
    """
    # 导入指数数据
    df_index = pd.read_csv(path, parse_dates=['candle_end_time'], encoding='gbk')
    df_index['指数涨跌幅'] = df_index['close'].pct_change()
    df_index = df_index[['candle_end_time', '指数涨跌幅']]
    df_index.dropna(subset=['指数涨跌幅'], inplace=True)
    df_index.rename(columns={'candle_end_time': '交易日期'}, inplace=True)

    if back_trader_start:
        df_index = df_index[df_index['交易日期'] >= pd.to_datetime(back_trader_start)]
    if back_trader_end:
        df_index = df_index[df_index['交易日期'] <= pd.to_datetime(back_trader_end)]

    df_index.sort_values(by=['交易日期'], inplace=True)
    df_index.reset_index(inplace=True, drop=True)

    return df_index


def merge_with_index_data(df, index_data, extra_fill_0_list=[]):
    """
    原始股票数据在不交易的时候没有数据。
    将原始股票数据和指数数据合并，可以补全原始股票数据没有交易的日期。
    :param df: 股票数据
    :param index_data: 指数数据
    :param extra_fill_0_list: 合并时需要填充为0的字段
    :return:
    """
    # ===将股票数据和上证指数合并，结果已经排序
    df = pd.merge(left=df, right=index_data, on='交易日期', how='right', sort=True, indicator=True)

    # ===对开、高、收、低、前收盘价价格进行补全处理
    # 用前一天的收盘价，补全收盘价的空值
    df['收盘价'].fillna(method='ffill', inplace=True)
    # 用收盘价补全开盘价、最高价、最低价的空值
    df['开盘价'].fillna(value=df['收盘价'], inplace=True)
    df['最高价'].fillna(value=df['收盘价'], inplace=True)
    df['最低价'].fillna(value=df['收盘价'], inplace=True)

    # 如果前面算过复权，复权价也做fillna
    if '收盘价_复权' in df.columns:
        df['收盘价_复权'].fillna(method='ffill', inplace=True)
        for col in ['开盘价_复权', '最高价_复权', '最低价_复权']:
            if col in df.columns:
                df[col].fillna(value=df['收盘价_复权'], inplace=True)

    # 补全前收盘价
    df['前收盘价'].fillna(value=df['收盘价'].shift(), inplace=True)

    # ===将停盘时间的某些列，数据填补为0
    fill_0_list = ['成交量', '成交额', '涨跌幅', '开盘买入涨跌幅'] + extra_fill_0_list
    df.loc[:, fill_0_list] = df[fill_0_list].fillna(value=0)

    # ===用前一天的数据，补全其余空值
    df.fillna(method='ffill', inplace=True)

    # ===去除上市之前的数据
    df = df[df['股票代码'].notnull()]

    # ===判断计算当天是否交易
    df['是否交易'] = 1
    df.loc[df['_merge'] == 'right_only', '是否交易'] = 0
    del df['_merge']
    df.reset_index(drop=True, inplace=True)
    return df


def transfer_to_period_data(df, po_df, period_type='M', extra_agg_dict={}, offset=0):
    """
    将日线数据转换为相应的周期数据
    :param df:原始数据
    :param po_df:从period_offset.csv载入的数据
    :param period_type:转换周期
    :param extra_agg_dict:
    :param offset:

    :return:
    """

    df['周期最后交易日'] = df['交易日期']

    agg_dict = {
        # 必须列
        '周期最后交易日': 'last',
        '股票代码': 'last',
        '股票名称': 'last',
        '是否交易': 'last',

        '开盘价': 'first',
        '最高价': 'max',
        '最低价': 'min',
        '收盘价': 'last',
        '成交额': 'sum',
        '流通市值': 'last',
        '总市值': 'last',
        '上市至今交易天数': 'last',

        '下日_是否交易': 'last',
        '下日_开盘涨停': 'last',
        '下日_是否ST': 'last',
        '下日_是否S': 'last',
        '下日_是否退市': 'last',
        '下日_开盘买入涨跌幅': 'last',

    }
    agg_dict = dict(agg_dict, **extra_agg_dict)
    # ===获取period、offset对应的周期表
    # _group为含负数的原始数据，用于把对应非交易日的涨跌幅设置为0
    po_df['_group'] = po_df[f'{period_type}_{offset}'].copy()
    # group为绝对值后的数据，用于对股票数据做groupby
    po_df['group'] = po_df['_group'].abs().copy()
    df = pd.merge(left=df, right=po_df[['交易日期', 'group', '_group']], on='交易日期', how='left')
    # 为了W53（周五买周三卖）这种有空仓日期的周期，把空仓日的涨跌幅设置为0
    df.loc[df['_group'] < 0, '涨跌幅'] = 0

    # ===对个股数据根据周期offset情况，进行groupby后，得到对应的nD/周线/月线数据
    period_df = df.groupby('group').agg(agg_dict)

    # 计算必须额外数据
    period_df['交易天数'] = df.groupby('group')['是否交易'].sum()
    period_df['市场交易天数'] = df.groupby('group')['是否交易'].count()
    # 计算其他因子
    # 计算周期资金曲线
    period_df['每天涨跌幅'] = df.groupby('group')['涨跌幅'].apply(lambda x: list(x))
    period_df['涨跌幅'] = df.groupby('group')['涨跌幅'].apply(lambda x: (x + 1).prod() - 1)
    period_df.rename(columns={'周期最后交易日': '交易日期'}, inplace=True)

    # 重置索引
    period_df.reset_index(drop=True, inplace=True)

    return period_df


# 计算涨跌停
def cal_zdt_price(df):
    """
    计算股票当天的涨跌停价格。在计算涨跌停价格的时候，按照严格的四舍五入。
    包含st股，但是不包含新股
    涨跌停制度规则:
        ---2020年8月23日
        非ST股票 10%
        ST股票 5%

        ---2020年8月24日至今
        普通非ST股票 10%
        普通ST股票 5%

        科创板（sh68） 20%（一直是20%，不受时间限制）
        创业板（sz30） 20%
        科创板和创业板即使ST，涨跌幅限制也是20%

        北交所（bj） 30%

    :param df: 必须得是日线数据。必须包含的字段：前收盘价，开盘价，最高价，最低价
    :return:
    """
    # 计算涨停价格
    # 普通股票
    cond = df['股票名称'].str.contains('ST')
    df['涨停价'] = df['前收盘价'] * 1.1
    df['跌停价'] = df['前收盘价'] * 0.9
    df.loc[cond, '涨停价'] = df['前收盘价'] * 1.05
    df.loc[cond, '跌停价'] = df['前收盘价'] * 0.95

    # 科创板 20%
    rule_kcb = df['股票代码'].str.contains('sh68')
    # 2020年8月23日之后涨跌停规则有所改变
    # 新规的创业板
    new_rule_cyb = (df['交易日期'] > pd.to_datetime('2020-08-23')) & df['股票代码'].str.contains('sz30')
    # 北交所条件
    cond_bj = df['股票代码'].str.contains('bj')

    # 科创板 & 创业板
    df.loc[rule_kcb | new_rule_cyb, '涨停价'] = df['前收盘价'] * 1.2
    df.loc[rule_kcb | new_rule_cyb, '跌停价'] = df['前收盘价'] * 0.8

    # 北交所
    df.loc[cond_bj, '涨停价'] = df['前收盘价'] * 1.3
    df.loc[cond_bj, '跌停价'] = df['前收盘价'] * 0.7

    # 四舍五入
    price_round = lambda x: float(Decimal(x + 1e-7).quantize(Decimal('1.00'), ROUND_HALF_UP))
    df['涨停价'] = df['涨停价'].apply(price_round)
    df['跌停价'] = df['跌停价'].apply(price_round)

    # 判断是否一字涨停
    df['一字涨停'] = False
    df.loc[df['最低价'] >= df['涨停价'], '一字涨停'] = True
    # 判断是否一字跌停
    df['一字跌停'] = False
    df.loc[df['最高价'] <= df['跌停价'], '一字跌停'] = True
    # 判断是否开盘涨停
    df['开盘涨停'] = False
    df.loc[df['开盘价'] >= df['涨停价'], '开盘涨停'] = True
    # 判断是否开盘跌停
    df['开盘跌停'] = False
    df.loc[df['开盘价'] <= df['跌停价'], '开盘跌停'] = True

    return df


# 计算策略评价指标
def strategy_evaluate(equity, select_stock):
    """
    :param equity:  每天的资金曲线
    :param select_stock: 每周期选出的股票
    :return:
    """

    # ===新建一个dataframe保存回测指标
    results = pd.DataFrame()

    # ===计算累积净值
    results.loc[0, '累积净值'] = round(equity['equity_curve'].iloc[-1], 2)

    # ===计算年化收益
    annual_return = (equity['equity_curve'].iloc[-1]) ** (
            '1 days 00:00:00' / (equity['交易日期'].iloc[-1] - equity['交易日期'].iloc[0]) * 365) - 1
    results.loc[0, '年化收益'] = str(round(annual_return * 100, 2)) + '%'

    # ===计算最大回撤，最大回撤的含义：《如何通过3行代码计算最大回撤》https://mp.weixin.qq.com/s/Dwt4lkKR_PEnWRprLlvPVw
    # 计算当日之前的资金曲线的最高点
    equity['max2here'] = equity['equity_curve'].expanding().max()
    # 计算到历史最高值到当日的跌幅，drowdwon
    equity['dd2here'] = equity['equity_curve'] / equity['max2here'] - 1
    # 计算最大回撤，以及最大回撤结束时间
    end_date, max_draw_down = tuple(equity.sort_values(by=['dd2here']).iloc[0][['交易日期', 'dd2here']])
    # 计算最大回撤开始时间
    start_date = equity[equity['交易日期'] <= end_date].sort_values(by='equity_curve', ascending=False).iloc[0][
        '交易日期']
    # 将无关的变量删除
    # equity.drop(['max2here', 'dd2here'], axis=1, inplace=True)
    results.loc[0, '最大回撤'] = format(max_draw_down, '.2%')
    results.loc[0, '最大回撤开始时间'] = str(start_date)
    results.loc[0, '最大回撤结束时间'] = str(end_date)

    # ===年化收益/回撤比：我个人比较关注的一个指标
    results.loc[0, '年化收益/回撤比'] = round(annual_return / abs(max_draw_down), 2)

    # ===统计每个周期
    if not select_stock.empty:
        results.loc[0, '盈利周期数'] = len(select_stock.loc[select_stock['选股下周期涨跌幅'] > 0])  # 盈利笔数
        results.loc[0, '亏损周期数'] = len(select_stock.loc[select_stock['选股下周期涨跌幅'] <= 0])  # 亏损笔数
        results.loc[0, '胜率'] = format(results.loc[0, '盈利周期数'] / len(select_stock), '.2%')  # 胜率
        results.loc[0, '每周期平均收益'] = format(select_stock['选股下周期涨跌幅'].mean(), '.2%')  # 每笔交易平均盈亏
        results.loc[0, '盈亏收益比'] = round(
            select_stock.loc[select_stock['选股下周期涨跌幅'] > 0]['选股下周期涨跌幅'].mean() / \
            select_stock.loc[select_stock['选股下周期涨跌幅'] <= 0]['选股下周期涨跌幅'].mean() * (-1), 2)  # 盈亏比
        results.loc[0, '单周期最大盈利'] = format(select_stock['选股下周期涨跌幅'].max(), '.2%')  # 单笔最大盈利
        results.loc[0, '单周期大亏损'] = format(select_stock['选股下周期涨跌幅'].min(), '.2%')  # 单笔最大亏损

        # ===连续盈利亏损
        results.loc[0, '最大连续盈利周期数'] = max(
            [len(list(v)) for k, v in
             itertools.groupby(np.where(select_stock['选股下周期涨跌幅'] > 0, 1, np.nan))])  # 最大连续盈利次数
        results.loc[0, '最大连续亏损周期数'] = max(
            [len(list(v)) for k, v in
             itertools.groupby(np.where(select_stock['选股下周期涨跌幅'] <= 0, 1, np.nan))])  # 最大连续亏损次数

    # ===每年、每月收益率
    equity.set_index('交易日期', inplace=True)
    year_return = equity[['涨跌幅']].resample(rule='A').apply(lambda x: (1 + x).prod() - 1)
    year_return['指数涨跌幅'] = equity[['指数涨跌幅']].resample(rule='A').apply(lambda x: (1 + x).prod() - 1)
    year_return['超额收益'] = year_return['涨跌幅'] - year_return['指数涨跌幅']

    def num2pct(x):
        return str(round(x * 100, 2)) + '%'

    year_return['涨跌幅'] = year_return['涨跌幅'].apply(num2pct)
    year_return['指数涨跌幅'] = year_return['指数涨跌幅'].apply(num2pct)
    year_return['超额收益'] = year_return['超额收益'].apply(num2pct)

    monthly_return = equity[['涨跌幅']].resample(rule='M').apply(lambda x: (1 + x).prod() - 1)
    monthly_return['指数涨跌幅'] = equity[['指数涨跌幅']].resample(rule='M').apply(lambda x: (1 + x).prod() - 1)
    monthly_return['超额收益'] = monthly_return['涨跌幅'] - monthly_return['指数涨跌幅']

    monthly_return['涨跌幅'] = monthly_return['涨跌幅'].apply(num2pct)
    monthly_return['指数涨跌幅'] = monthly_return['指数涨跌幅'].apply(num2pct)
    monthly_return['超额收益'] = monthly_return['超额收益'].apply(num2pct)

    return results.T, year_return, monthly_return


def create_empty_data(index_data, period, offset, po_df):
    empty_df = index_data[['交易日期']].copy()
    empty_df['涨跌幅'] = 0.0
    empty_df['周期最后交易日'] = empty_df['交易日期']
    agg_dict = {'周期最后交易日': 'last'}
    po_df['group'] = po_df[f'{period}_{offset}'].abs().copy()
    group = po_df[['交易日期', 'group']].copy()
    empty_df = pd.merge(left=empty_df, right=group, on='交易日期', how='left')
    empty_period_df = empty_df.groupby('group').agg(agg_dict)
    empty_period_df['每天涨跌幅'] = empty_df.groupby('group')['涨跌幅'].apply(lambda x: list(x))
    # 删除没交易的日期
    empty_period_df.dropna(subset=['周期最后交易日'], inplace=True)

    empty_period_df['选股下周期每天涨跌幅'] = empty_period_df['每天涨跌幅'].shift(-1)
    empty_period_df.dropna(subset=['选股下周期每天涨跌幅'], inplace=True)

    # 填仓其他列
    empty_period_df['股票数量'] = 0
    empty_period_df['买入股票代码'] = 'empty'
    empty_period_df['买入股票名称'] = 'empty'
    empty_period_df['选股下周期涨跌幅'] = 0.0
    empty_period_df.rename(columns={'周期最后交易日': '交易日期'}, inplace=True)

    empty_period_df.set_index('交易日期', inplace=True)

    empty_period_df = empty_period_df[
        ['股票数量', '买入股票代码', '买入股票名称', '选股下周期涨跌幅', '选股下周期每天涨跌幅']]
    return empty_period_df


def equity_to_csv(equity, strategy_name, period, offset, select_stock_num, folder_path):
    """
    输出策略轮动对应的文件
    :param equity: 策略资金曲线
    :param strategy_name: 策略名称
    :param period: 周期
    :param offset: offset
    :param select_stock_num: 选股数
    :param folder_path: 输出路径
    :return:
    """
    period_dict = {  # 周期对应的字典，暂时不兼容工作月的策略
        'W': 'week',
        'M': 'natural_month',
    }
    if type(period) == int:
        period_dict[period] = period
    if period not in period_dict:
        period_dict[period] = re.sub(r'(\d+)M', r'\1month', re.sub(r'(\d+)W', r'\1week', period))
    to_csv_path = folder_path + f'/{strategy_name}_{period_dict[period]}_{offset}_{select_stock_num}.csv'
    equity['策略名称'] = strategy_name + '_' + str(period_dict[period]) + '_' + str(select_stock_num)
    pd.DataFrame(columns=['数据由邢不行整理，对数据字段有疑问的，可以直接微信私信邢不行，微信号：xbx297']).to_csv(
        to_csv_path,
        encoding='gbk',
        index=False)
    equity = equity[['交易日期', '策略名称', '持有股票代码', '涨跌幅', 'equity_curve', '指数涨跌幅', 'benchmark']]
    equity.to_csv(to_csv_path, encoding='gbk', index=False, mode='a')


def create_daily_result(path, name, period, offset, count):
    # 创建每日选股的文件
    period_dict = {  # 周期对应的字典，暂时不兼容工作月的策略
        'W': 'week',
        'M': 'natural_month',
    }
    if type(period) == int:
        period_dict[period] = period
    if period not in period_dict:
        period_dict[period] = re.sub(r'(\d+)M', r'\1month', re.sub(r'(\d+)W', r'\1week', period))

    # 构建保存的路径
    file_path = path + '/%s_%s_%s_%s.csv' % (name, period_dict[period], offset, count)
    if not os.path.exists(file_path):
        res_df = pd.DataFrame(columns=['选股日期', '股票代码', '股票名称', '选股排名'])
        res_df.to_csv(file_path, encoding='gbk', index=False)


def save_select_result(path, new_res, name, period, offset, count, signal=1):
    """
    保存选股数据的最新结果
    :param path: 保存结果的文件夹路劲
    :param new_res: 选股数据
    :param name: 策略名称
    :param period: 持仓周期
    :param offset: offset
    :param count: 选股数据
    :param signal: 择时信号
    :return:
    """
    period_dict = {  # 周期对应的字典，暂时不兼容工作月的策略
        'W': 'week',
        'M': 'natural_month',
    }
    if type(period) == int:
        period_dict[period] = period
    if period not in period_dict:
        period_dict[period] = re.sub(r'(\d+)M', r'\1month', re.sub(r'(\d+)W', r'\1week', period))

    new_res = new_res[['交易日期', '股票代码', '股票名称', '选股排名']].rename(columns={'交易日期': '选股日期'})
    if signal != 1:
        new_res = pd.DataFrame(columns=['选股日期', '股票代码', '股票名称', '选股排名'])

    # 构建保存的路径
    file_path = path + '/%s_%s_%s_%s.csv' % (name, period_dict[period], offset, count)
    # 申明历史选股结果的变量
    res_df = pd.DataFrame()

    # 如果有历史结果，则读取历史结果
    if os.path.exists(file_path):
        res_df = pd.read_csv(file_path, encoding='gbk', parse_dates=['选股日期'])
        # 有新产生持仓，就把历史结果里的相同日期去掉
        if not new_res.empty:
            res_df = res_df[res_df['选股日期'] < new_res['选股日期'].min()]
    # 将历史选股结果与最新选股结果合并
    res_df = pd.concat([res_df, new_res], ignore_index=True)
    # 清洗数据，保存结果
    res_df.drop_duplicates(subset=['选股日期', '股票代码'], keep='last', inplace=True)
    res_df.sort_values(by=['选股日期', '选股排名'], inplace=True)
    res_df.to_csv(file_path, encoding='gbk', index=False)


def _factors_linear_regression(data, factor, neutralize_list, industry=None):
    """
    使用线性回归对目标因子进行中性化处理，此方法外部不可直接调用。
    :param data: 股票数据
    :param factor: 目标因子
    :param neutralize_list:中性化处理变量list
    :param industry: 行业字段名称，默认为None
    :return: 中性化之后的数据
    """

    train_col = []
    train_col += neutralize_list

    lrm = LinearRegression(fit_intercept=True)  # 创建线性回归模型
    if industry:  # 如果需要对行业进行中性化，将行业的列名加入到neutralize_list中
        # 获取一下当周期有什么行业，申万一级行业发生过拆分，所以需要考虑
        ind_list = list(data[industry].unique())
        ind_list = ['所属行业_' + ind for ind in ind_list]

        industry_cols = [col for col in data.columns if '所属行业' in col]
        for col in industry_cols:
            if col not in train_col:
                if col in ind_list:
                    train_col.append(col)
    train = data[train_col].copy()  # 输入变量
    label = data[[factor]].copy()  # 预测变量
    lrm.fit(train, label)  # 线性拟合
    predict = lrm.predict(train)  # 输入变量进行预测
    data[factor + '_中性'] = label.values - predict  # 计算残差
    return data


def factor_neutralization(data, factor, neutralize_list, industry=None):
    """
    使用线性回归对目标因子进行中性化处理，此方法可以被外部调用。
    :param data: 股票数据
    :param factor: 目标因子
    :param neutralize_list:中性化处理变量list
    :param industry: 行业字段名称，默认为None
    :return: 中性化之后的数据
    """
    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[factor] + neutralize_list, how='any')
    if industry:  # 果需要对行业进行中性化，先构建行业哑变量
        # 剔除中性化所涉及的字段中，包含inf、-inf、nan的部分
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[industry], how='any')
        # 对行业进行哑变量处理
        ind = df[industry]
        ind = pd.get_dummies(ind, columns=[industry], prefix='所属行业',
                             prefix_sep="_", dummy_na=False, drop_first=False)
        """
        drop_first=True会导致某一行业的的哑变量被删除，这样的做的目的是为了消除行业间的多重共线性
        详见：https://www.learndatasci.com/glossary/dummy-variable-trap/

        2023年6月25日起
        不再使用drop_first=True，而指定一个行业直接删除，避免不同的周期删除不同的行业。
        """
        # 删除一个行业，原因如上提到的drop_first
        ind.drop(columns=['所属行业_综合'], inplace=True)
    else:
        ind = pd.DataFrame()
    df = pd.concat([df, ind], axis=1)

    df = df.groupby(['交易日期']).apply(_factors_linear_regression, factor=factor,
                                        neutralize_list=neutralize_list, industry=industry)

    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['交易日期', '股票代码'], inplace=True)
    return df


def calc_period_offset(stg_file):
    """
    读取所有策略的period和offset
    :param stg_file: 策略文件list，通过get_file_in_folder获取
    :return: 所有策略的所有period和offset，如{'W': [0, 1, 2, 5], '2W': [0, 1], '4W': [0, 2]}
    """
    period_dict = {}
    for file in stg_file:
        cls = __import__('program.选股策略.%s' % file, fromlist=('',))
        if cls.period not in period_dict:
            period_dict[cls.period] = []
        try:
            period_dict[cls.period] += cls.offset
        except:
            period_dict[cls.period] += [0]
        period_dict[cls.period] = list(set(period_dict[cls.period]))

    return period_dict


def merge_offset(equity_list, index_data):
    """
    合并所有offset的策略资金曲线
    :param equity_list: 各offset的资金曲线list
    :param index_data: 指数
    :return: equity_df, equity_df_notiming：合并完的资金曲线数据,合并完的未择时资金曲线
    """
    # 合并equity_list中所有资金曲线，填充空值，因为不同offset的起始结束日不同，所以肯定有空值
    _equity_df = pd.concat(equity_list, axis=1, join='outer')
    _equity_df.fillna(method='ffill', inplace=True)
    _equity_df.fillna(value=1, inplace=True)

    # 通过最大最小的时间，从index取出需要画图的这段，算完banchmark
    equity_df = index_data[
        (index_data['交易日期'] >= _equity_df.index.min()) & (index_data['交易日期'] <= _equity_df.index.max())].copy()
    equity_df.set_index('交易日期', inplace=True)
    equity_df['benchmark'] = (equity_df['指数涨跌幅'] + 1).cumprod()
    # 合并资金曲线，通过遍历择时和不择时区分两个
    equity_col = _equity_df.columns.unique().to_list()
    for each_col in equity_col:
        equity_df[each_col] = _equity_df[[each_col]].mean(axis=1)
    # 把交易日期变回非index的列
    equity_df.reset_index(drop=False, inplace=True)
    # 资金曲线反推的时候，需要前面加一行，否则第一个涨跌幅算不出
    equity_df = pd.concat([pd.DataFrame([{'equity_curve': 1}]), equity_df], ignore_index=True)
    equity_df['涨跌幅'] = equity_df['equity_curve'] / equity_df['equity_curve'].shift() - 1
    equity_df.drop([0], axis=0, inplace=True)
    if len(equity_col) > 1:
        # 带择时时，需要多算一遍
        equity_df_notiming = equity_df[['交易日期', '指数涨跌幅', 'benchmark', 'equity_curve_notiming']].copy()
        equity_df_notiming.rename(columns={'equity_curve_notiming': 'equity_curve'}, inplace=True)
        equity_df_notiming = pd.concat([pd.DataFrame([{'equity_curve': 1}]), equity_df_notiming], ignore_index=True)
        equity_df_notiming['涨跌幅'] = equity_df_notiming['equity_curve'] / equity_df_notiming[
            'equity_curve'].shift() - 1
        equity_df_notiming.drop([0], axis=0, inplace=True)
        equity_df = equity_df[['交易日期', '涨跌幅', 'equity_curve', '指数涨跌幅', 'benchmark']]
        equity_df_notiming = equity_df_notiming[['交易日期', '涨跌幅', 'equity_curve', '指数涨跌幅', 'benchmark']]
        return equity_df, equity_df_notiming

    else:
        # 不带择时
        equity_df = equity_df[['交易日期', '涨跌幅', 'equity_curve', '指数涨跌幅', 'benchmark']]
        return equity_df, pd.DataFrame()


def is_in_current_period(period, offset, index_data, period_and_offset_df):
    """
    判断指数文件的最后一行日期是否命中周期offset选股日（选股日定义：持仓周期最后一日，尾盘需要卖出持仓，次日早盘买入新的标的）
    :return: True选股日，False非选股日
    """
    index_lastday = index_data['交易日期'].iloc[-1]
    # 把周期数据转为选股日标记数据
    period_and_offset_df[f'{period}_{offset}_判断'] = period_and_offset_df[f'{period}_{offset}'].abs().diff().shift(-1)
    """
             交易日期  W_0  W_1  W_2  W_0_判断
        0  2005-01-05  0.0  0.0  0.0     0.0
        1  2005-01-06  0.0  0.0  0.0     0.0
        2  2005-01-07  0.0  0.0  0.0     1.0
        3  2005-01-10  1.0  0.0  0.0     0.0
        4  2005-01-11  1.0  1.0  0.0     0.0
        5  2005-01-12  1.0  1.0  1.0     0.0
        6  2005-01-13  1.0  1.0  1.0     0.0
        7  2005-01-14  1.0  1.0  1.0     1.0
        8  2005-01-17  2.0  1.0  1.0     0.0
    """
    if period_and_offset_df.loc[period_and_offset_df['交易日期'] == index_lastday, f'{period}_{offset}_判断'].iloc[-1] == 1:
        # 选股日
        return True
    else:
        # 非选股日
        return False


def read_period_and_offset_file(file_path):
    """
    载入周期offset文件
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='gbk', parse_dates=['交易日期'], skiprows=1)
        return df
    else:
        print(f'文件{file_path}不存在，请获取period_offset.csv文件后再试')
        raise FileNotFoundError('文件不存在')


def get_trade_info(_df, open_times, close_times, buy_method):
    """
    获取每一笔的交易信息
    :param _df:算完复权的基础价格数据
    :param open_times:买入日的list
    :param close_times:卖出日的list
    :param buy_method:同config.py中设定买入股票的方法，即在什么时候买入
    :return: df:'买入日期', '卖出日期', '买入价', '卖出价', '收益率',在个股结果中展示
    """

    df = pd.DataFrame(columns=['买入日期', '卖出日期'])
    df['买入日期'] = open_times
    df['卖出日期'] = close_times
    # 买入的价格合并
    df = pd.merge(left=df, right=_df[
        ['交易日期', f'{buy_method.replace("价", "")}价_复权', f'{buy_method.replace("价", "")}价']],
                  left_on='买入日期',
                  right_on='交易日期',
                  how='left')
    # 卖出的价格合并
    df = pd.merge(left=df, right=_df[['交易日期', '收盘价_复权', '收盘价']], left_on='卖出日期', right_on='交易日期',
                  how='left')
    # 展示的买入卖出价为非复权价
    df.rename(columns={f'{buy_method.replace("价", "")}价': '买入价', '收盘价': '卖出价'}, inplace=True)
    # 收益率用复权价算
    df['收益率'] = df['收盘价_复权'] / df[f'{buy_method.replace("价", "")}价_复权'] - 1
    # 将收益率转为为百分比格式
    df['收益率'] = df['收益率'].apply(lambda x: str(round(100 * x, 2)) + '%')
    df = df[['买入日期', '卖出日期', '买入价', '卖出价', '收益率']]
    return df


def merge_timing_data(rtn, rtn_notiming, year_return, year_return_notiming, month_return, month_return_notiming):
    # 合并带择时后的信息，用于统一print
    rtn.rename(columns={0: '带择时'}, inplace=True)
    rtn_notiming.rename(columns={0: '原策略'}, inplace=True)
    rtn = pd.concat([rtn_notiming, rtn], axis=1)
    year_return = pd.merge(left=year_return_notiming, right=year_return[['涨跌幅', '超额收益']],
                           left_index=True, right_index=True, how='outer', suffixes=('', '_(带择时)'))
    year_return = year_return[['涨跌幅', '涨跌幅_(带择时)', '指数涨跌幅', '超额收益', '超额收益_(带择时)']]
    month_return = pd.merge(left=month_return_notiming, right=month_return[['涨跌幅', '超额收益']],
                            left_index=True, right_index=True, how='outer', suffixes=('', '_(带择时)'))
    month_return = month_return[['涨跌幅', '涨跌幅_(带择时)', '指数涨跌幅', '超额收益', '超额收益_(带择时)']]
    return rtn, year_return, month_return


def auto_offset(period):
    res = [0]
    # 判断指定的period是否为int类型
    if isinstance(period, int):
        # 如果是int类型，则有int个offset
        res = list(range(0, period))
    # 判断指定的period是否为str类型
    elif isinstance(period, str):
        # 判断period中是否包含W
        if ('W' in period.upper()):
            # 如果period只有W，即 period == 'W'
            if len(period) == 1:
                res = [0, 1, 2, 3, 4]
            # 如果period为N个W，比如 period == '2W'，则有两个offset
            else:
                res = list(range(0, int(period[:-1])))
        # 判断period中是否包含M
        elif 'M' in period.upper():
            # 如果period == 'M'
            if len(period) == 1:
                res = [0]
            # 如果period 为 N个M，比如 period == '2M'，则有两个offset
            else:
                res = list(range(0, int(period[:-1])))

    return res
