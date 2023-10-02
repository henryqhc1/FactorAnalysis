'''
2023分享会
author: 邢不行
微信: xbx9585
'''
import numpy as np
import pandas as pd
import scipy
from program.Functions import *
import math


def float_num_process(num, return_type=float, keep=2, max=5):
    """
    针对绝对值小于1的数字进行特殊处理，保留非0的N位（N默认为2，即keep参数）
    输入  0.231  输出  0.23
    输入  0.0231  输出  0.023
    输入  0.00231  输出  0.0023
    如果前面max个都是0，直接返回0.0
    :param num: 输入的数据
    :param return_type: 返回的数据类型，默认是float
    :param keep: 需要保留的非零位数
    :param max: 最长保留多少位
    :return:
        返回一个float或str
    """

    # 如果输入的数据是0，直接返回0.0
    if num == 0.:
        return 0.0

    # 绝对值大于1的数直接保留对应的位数输出
    if abs(num) > 1:
        return round(num, keep)
    # 获取小数点后面有多少个0
    zero_count = -int(math.log10(abs(num)))
    # 实际需要保留的位数
    keep = min(zero_count + keep, max)

    # 如果指定return_type是float，则返回float类型的数据
    if return_type == float:
        return round(num, keep)
    # 如果指定return_type是str，则返回str类型的数据
    else:
        return str(round(num, keep))


def get_factor_by_period(folder, period, offset_list, target, need_shift, factor_cls):
    '''
    读取数据的函数
    :param folder: 数据所在的文件夹路径
    :param period: 配置的offset
    :param offset_list: 根据period生成的offset列表
    :param target: 目标列名
    :param need_shift: 目标列是否需要shift
    :param factor_cls: 选股策略的因子库的类
    :return:
        返回读取到的所有数据
    '''

    df_list = []
    # 遍历offset读入数据
    for offset in offset_list:
        # 指定路径
        file_path = folder + f'all_stock_data_{period}_{offset}.pkl'
        if not os.path.exists(file_path):
            print(f'offset_{offset}文件缺失，可能会影响测试结果')
            continue
        # 读入数据
        df = pd.read_pickle(file_path)
        # 筛选股票并计算排名
        df = factor_cls.filter_stock(df)
        # target列是否需要shift
        if need_shift:
            df['下周期_' + target] = df.groupby('股票代码').apply(lambda x: x[target].shift()).reset_index(0)[target]
        # 加入offset列
        df['offset'] = offset
        df_list.append(df)
    # 合并数据
    all_df = pd.concat(df_list, ignore_index=True)

    return all_df


def offset_grouping(df, factor, bins):
    '''
    分组函数
    :param df: 原数据
    :param factor: 因子名
    :param bins: 分组的数量
    :return:
        返回一个df数据，包含groups列
    '''

    # 根据factor计算因子的排名
    df['因子_排名'] = df.groupby(['交易日期'])[factor].rank(ascending=True, method='first')
    # 根据因子的排名进行分组
    df['groups'] = df.groupby(['交易日期'])['因子_排名'].transform(
        lambda x: pd.qcut(x, q=bins, labels=range(1, bins + 1), duplicates='drop'))

    # 这里不需要判断某个周期的股票数量大于bins，因为之前在处理limit时已经处理过这个问题

    return df


def IC_analysis(df, factor, target):
    '''
    计算IC等一系列指标
    :param df: 数据
    :param factor: 因子列名：测试的因子名称
    :param target: 目标列名：计算IC时的下周期数据
    :return:
        返回IC数据、IC字符串
    '''

    print('正在进行因子IC分析...')

    # 计算IC并处理数据
    corr = df.groupby('交易日期').apply(lambda x: x[factor].corr(x[target], method='spearman')).to_frame()
    corr = corr.rename(columns={0: 'RankIC'}).reset_index()
    # 计算IC的累加值；注意：因为我们考虑了每个offset，所以这边为了
    corr['累计RankIC'] = corr['RankIC'].cumsum() / (df['offset'].max() + 1)

    # ===计算IC的统计值，并进行约等
    # =IC均值
    IC_mean = float_num_process(corr['RankIC'].mean())
    # =IC标准差
    IC_std = float_num_process(corr['RankIC'].std())
    # =ICIR
    ICIR = float_num_process(IC_mean / IC_std)
    # =IC胜率
    # 如果累计IC为正，则计算IC为正的比例
    if corr['累计RankIC'].iloc[-1] > 0:
        IC_ratio = str(float_num_process((corr['RankIC'] > 0).sum() / len(corr) * 100)) + '%'
    # 如果累计IC为负，则计算IC为负的比例
    else:
        IC_ratio = str(float_num_process((corr['RankIC'] < 0).sum() / len(corr) * 100)) + '%'

    # 将上述指标合成一个字符串，加入到IC图中
    IC_info = f'IC均值：{IC_mean}，IC标准差：{IC_std}，ICIR：{ICIR}，IC胜率：{IC_ratio}'

    return corr, IC_info


def get_corr_month(corr):
    '''
    生成IC月历
    :param corr: IC数据
    :return:
        返回IC月历的df数据
    '''

    print('正在进行IC月历计算...')

    # resample到月份数据
    corr['交易日期'] = pd.to_datetime(corr['交易日期'])
    corr.set_index('交易日期', inplace=True)
    corr_month = corr.resample('M').agg({'RankIC': 'mean'})
    corr_month.reset_index(inplace=True)
    # 提取出年份和月份
    corr_month['年份'] = corr_month['交易日期'].map(lambda x: str(x)[:4])
    corr_month['月份'] = corr_month['交易日期'].map(lambda x: str(x)[5:7])
    # 将年份月份设置为index，在将月份unstack为列
    corr_month = corr_month.set_index(['年份', '月份'])['RankIC']
    corr_month = corr_month.unstack('月份')
    # 计算各月平均的IC
    corr_month.loc['各月平均', :] = corr_month.mean(axis=0)
    # 按年份大小排名
    corr_month = corr_month.sort_index(ascending=False)

    return corr_month


def group_analysis(df, next_ret, b_rate, s_rate):
    """
    针对分组数据进行分析，给出分组的资金曲线、分箱图以及各分组的未来资金曲线
    :param df: 输入的数据
    :param next_ret: 未来涨跌幅的list
    :param b_rate: 买入手续费用
    :param s_rate: 卖出手续费用
    :return:
        返回分组资金曲线、分箱图、分组持仓走势数据
    """

    print('正在进行因子分组分析...')

    # 由于会对原始的数据进行修正，所以需要把数据copy一份
    temp = df.copy()
    time_df = pd.DataFrame(sorted(df['交易日期'].unique()), columns=['交易日期'])

    # 将持仓周期的中位数当做标准的持仓周期数
    temp['持仓周期'] = temp[next_ret].apply(lambda x: len(x))
    hold_nums = int(temp['持仓周期'].mode())
    # 实际持仓＜标准周期数，用0补全
    temp.loc[temp['持仓周期'] < hold_nums, next_ret] = temp[next_ret].apply(lambda x: x + [0] * (hold_nums - len(x)))
    # 实际持仓＞标准周期数，截取到当前值即可
    temp.loc[temp['持仓周期'] > hold_nums, next_ret] = temp[next_ret].apply(lambda x: x[:hold_nums])

    # temp['下周期每天涨跌幅'] = temp[next_ret].apply(lambda x: [(1 + x[0]) * (1 - b_rate) - 1] + x[0:-1] + [(1 + x[-1]) * (1 - s_rate) - 1])
    # group_nv = temp.groupby(['交易日期', 'groups']).apply(lambda x: np.array(x['下周期每天涨跌幅']).mean()).reset_index()
    # group_nv = group_nv.sort_values(by='交易日期').reset_index(drop=True)

    # 计算下周期每天的净值，并扣除手续费得到下周期的实际净值
    temp['下周期每天净值'] = temp[next_ret].apply(lambda x: (np.array(x) + 1).cumprod())
    free_rate = (1 - b_rate) * (1 - s_rate)
    temp['下周期净值'] = temp['下周期每天净值'].apply(lambda x: x[-1] * free_rate)

    # 按照offset分组，计算各个offset的分组资金曲线
    groups = temp.groupby('offset')

    nv_list = []
    for offset, group in groups:
        # 当前offset所有分组每个周期的净值
        group_nv = group.groupby(['交易日期', 'groups'])['下周期净值'].mean().reset_index()
        group_nv = group_nv.sort_values(by='交易日期').reset_index(drop=True)
        # 将每个周期的净值-1，得到每个周期的涨跌幅
        group_nv['下周期涨跌幅'] = group_nv['下周期净值'] - 1
        # 计算每个分组的累计净值
        group_nv['净值'] = group_nv.groupby('groups')['下周期涨跌幅'].apply(lambda x: (x + 1).cumprod())

        # 将所有的数据合并上全量的时间，并用前值填充nan
        group_nv = group_nv.sort_values(by=['groups', '交易日期']).reset_index(drop=True)
        group_nv = group_nv.groupby('groups').apply(
            lambda x: pd.merge(time_df, x, 'left', '交易日期').fillna(method='ffill'))
        nv_list.append(group_nv)

    # 将所有offset的分组资金曲线数据合并
    nv_df = pd.concat(nv_list, ignore_index=True)
    # 计算当前数据有多少个分组
    bins = nv_df['groups'].max()
    # 计算不同offset的每个分组的平均净值
    group_curve = nv_df.groupby(['交易日期', 'groups'])['净值'].mean().reset_index()
    # 将数据按照展开
    group_curve = group_curve.set_index(['交易日期', 'groups']).unstack().reset_index()
    # 重命名数据列
    group_cols = ['交易日期'] + [f'第{i}组' for i in range(1, bins + 1)]
    group_curve.columns = group_cols

    # 计算多空净值走势
    # 获取第一组的涨跌幅数据
    first_group_ret = group_curve['第1组'].pct_change()
    first_group_ret.fillna(value=group_curve['第1组'].iloc[0] - 1, inplace=True)
    # 获取最后一组的涨跌幅数据
    last_group_ret = group_curve[f'第{bins}组'].pct_change()
    last_group_ret.fillna(value=group_curve[f'第{bins}组'].iloc[0] - 1, inplace=True)
    # 判断到底是多第一组空最后一组，还是多最后一组空第一组
    if group_curve['第1组'].iloc[-1] > group_curve[f'第{bins}组'].iloc[-1]:
        ls_ret = (first_group_ret - last_group_ret) / 2
    else:
        ls_ret = (last_group_ret - first_group_ret) / 2
    # 计算多空净值曲线
    group_curve['多空净值'] = (ls_ret + 1).cumprod()

    # 计算绘制分箱所需要的数据
    group_value = group_curve[-1:].T[1:].reset_index()
    group_value.columns = ['分组', '净值']

    # 计算各分组在持仓内的每天收益
    group_hold_value = pd.DataFrame(temp.groupby('groups')['下周期每天净值'].mean()).T
    # 所有分组的第一天都是从1开始的
    for col in group_hold_value.columns:
        group_hold_value[col] = group_hold_value[col].apply(lambda x: [1] + list(x))
    # 将未来收益从list展开成逐行的数据
    group_hold_value = group_hold_value.explode(list(group_hold_value.columns)).reset_index(drop=True).reset_index()
    # 重命名列
    group_cols = ['时间'] + [f'第{i}组' for i in range(1, bins + 1)]
    group_hold_value.columns = group_cols

    # 返回数据：分组资金曲线、分组净值、分组持仓走势
    return group_curve, group_value, group_hold_value


def style_analysis(df, factor):
    '''
    计算因子的风格暴露
    :param df: df数据，包含因子列和风格列
    :param factor: 因子列
    :return:
        返回因子的风格暴露的数据
    '''

    print('正在进行因子风格暴露分析...')

    # 取出风格列，格式：以 风格_ 开头
    style_cols = [col for col in df.columns if col.startswith('风格因子_')]

    if len(style_cols) == 0:
        return pd.DataFrame()


    # 计算因子与风格的相关系数
    style_corr = df[[factor] + style_cols].corr(method='spearman').iloc[0, 1:].to_frame().reset_index()
    # 整理数据
    style_corr = style_corr.rename(columns={'index': '风格', factor: '相关系数'})
    style_corr['风格'] = style_corr['风格'].map(lambda x: x.split('_')[1])

    return style_corr


def industry_analysis(df, factor, target, industry_col, industry_name_change):
    '''
    计算分行业的IC
    :param df: 原始数据
    :param factor: 因子列
    :param target: 目标列
    :param industry_col: 配置的行业列名
    :return:
        返回各个行业的RankIC数据、占比数据
    '''

    print('正在进行因子行业分析...')

    def get_industry_data(temp):
        '''
        计算分行业IC、占比
        :param temp: 每个行业的数据
        :return:
            返回IC序列的均值、第一组占比、最后一组占比
        '''
        # 计算每个行业的IC值
        ic = temp.groupby('交易日期').apply(lambda x: x[factor].corr(x[target], method='spearman'))
        # 计算每个行业的第一组的占比和最后一组的占比
        part_min_group = temp.groupby('交易日期').apply(lambda x: (x['groups'] == min_group).sum())
        part_max_group = temp.groupby('交易日期').apply(lambda x: (x['groups'] == max_group).sum())
        part_min_group = part_min_group / all_min_group
        part_max_group = part_max_group / all_max_group

        return [ic.mean(), part_min_group.mean(), part_max_group.mean()]

    # 替换行业名称
    df[industry_col] = df[industry_col].replace(industry_name_change)
    # 获取以因子分组第一组和最后一组的数量
    min_group, max_group = df['groups'].min(), df['groups'].max()
    all_min_group = df.groupby('交易日期').apply(lambda x: (x['groups'] == min_group).sum())
    all_max_group = df.groupby('交易日期').apply(lambda x: (x['groups'] == max_group).sum())
    # 以行业分组计算IC及占比，并处理数据
    industry_data = df.groupby(industry_col).apply(get_industry_data).to_frame().reset_index()
    # 取出IC数据、行业占比_第一组数据、行业占比_最后一组数据
    industry_data['RankIC'] = industry_data[0].map(lambda x: x[0])
    industry_data['行业占比_第一组'] = industry_data[0].map(lambda x: x[1])
    industry_data['行业占比_最后一组'] = industry_data[0].map(lambda x: x[2])
    # 处理数据
    industry_data.drop(0, axis=1, inplace=True)
    # 以IC排序
    industry_data.sort_values('RankIC', ascending=False, inplace=True)

    return industry_data


def market_value_analysis(df, factor, target, market_value, bins=10):
    '''
    计算分市值的IC数据
    :param df: 原数据
    :param factor: 因子名
    :param target: 目标名
    :param market_value: 配置的市值列名
    :param bins: 分组的数量
    :return:
        返回各个市值分组的IC、占比数据
    '''

    print('正在进行因子市值分析...')

    # 先对市值数据进行排名以及分组
    df['市值_排名'] = df.groupby(['交易日期'])[market_value].rank(ascending=True, method='first')
    df['市值分组'] = df.groupby(['交易日期'])['市值_排名'].transform(
        lambda x: pd.qcut(x, q=bins, labels=range(1, bins + 1), duplicates='drop'))

    def get_market_value_data(temp):
        '''
        计算分市值IC、占比
        :param temp: 每个市值分组的数据
        :return:
            返回IC序列的均值、第一组占比、最后一组占比
        '''
        # 计算每个市值分组的IC值
        ic = temp.groupby('交易日期').apply(lambda x: x[factor].corr(x[target], method='spearman'))
        # 计算每个市值分组的第一组的占比和最后一组的占比
        part_min_group = temp.groupby('交易日期').apply(lambda x: (x['groups'] == min_group).sum())
        part_max_group = temp.groupby('交易日期').apply(lambda x: (x['groups'] == max_group).sum())
        part_min_group = part_min_group / all_min_group
        part_max_group = part_max_group / all_max_group

        return [ic.mean(), part_min_group.mean(), part_max_group.mean()]

    # 获取以因子分组第一组和最后一组的数量
    min_group, max_group = df['groups'].min(), df['groups'].max()
    all_min_group = df.groupby('交易日期').apply(lambda x: (x['groups'] == min_group).sum())
    all_max_group = df.groupby('交易日期').apply(lambda x: (x['groups'] == max_group).sum())
    # 根据市值分组计算IC及占比，并处理数据
    market_value_data = df.groupby('市值分组').apply(get_market_value_data).to_frame().reset_index()
    # 取出IC数据、市值占比_第一组数据、市值占比_最后一组数据
    market_value_data['RankIC'] = market_value_data[0].map(lambda x: x[0])
    market_value_data['市值占比_第一组'] = market_value_data[0].map(lambda x: x[1])
    market_value_data['市值占比_最后一组'] = market_value_data[0].map(lambda x: x[2])
    # 处理数据
    market_value_data.drop(0, axis=1, inplace=True)
    # 以市值分组大小排序
    market_value_data.sort_index(ascending=True, inplace=True)

    return market_value_data
