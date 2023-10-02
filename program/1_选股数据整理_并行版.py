"""
2023分享会
author: 邢不行
微信: xbx9585
选股策略框架
"""
import os.path
from datetime import datetime
import time
import pandas as pd
from joblib import Parallel, delayed
from Config import *
from program.Function_fin import *
from program.Functions import *


# ===循环读取并且合并
def calculate_by_stock(code):
    """
    整理数据核心函数
    :param code: 股票代码
    :return: 一个包含该股票所有历史数据的DataFrame
    """
    print(code, '开始计算')
    # =读入股票数据
    path = stock_data_path + code
    df = pd.read_csv(path, encoding='gbk', skiprows=1, parse_dates=['交易日期'])

    # =计算涨跌幅
    df['涨跌幅'] = df['收盘价'] / df['前收盘价'] - 1
    df['开盘买入涨跌幅'] = df['收盘价'] / df['开盘价'] - 1  # 为之后开盘买入做好准备
    df['换手率'] = df['成交额'] / df['流通市值']

    # =计算复权价：计算所有因子当中用到的价格，都使用复权价
    df = cal_fuquan_price(df)

    # =计算涨跌停价格
    df = cal_zdt_price(df)
    # =计算交易天数
    df['上市至今交易天数'] = df.index.astype('int') + 1

    # 根据subset_day把数据切小，让因子运算变快
    df = df[df['交易日期'] >= subset_day]

    # 转换周期时需要额外处理的字段
    exg_dict = {}  # 在转换周期时使用
    # 和指数合并时需要额外处理的字段
    fill_0_list = ['换手率']  # 在和上证指数合并时使用。

    # =合并指数之前的操作
    for strategy in stg_list:  # 遍历每个策略
        df, exg_dict, fill_0_list = strategy.before_merge_index(df, exg_dict, fill_0_list)
    fill_0_list = list(set(fill_0_list))

    # =将股票和上证指数合并，补全停牌的日期，新增数据"是否交易"、"指数涨跌幅"
    df = merge_with_index_data(df, index_data, fill_0_list)
    # =股票退市时间小于指数开始时间，就会出现空值
    if df.empty:
        return pd.DataFrame()

    # =导入财务数据，将个股数据和财务数据合并，并计算需要的财务指标的衍生指标
    df, fin_df, fin_raw_df = merge_with_finance_data(df, code[:-4], fin_path, add_fin_cols, exg_dict, flow_fin_cols,
                                                     cross_fin_cols)

    # =遍历每个策略，计算其需要合并的其他数据
    for strategy in stg_list:
        df, exg_dict = strategy.merge_single_stock_file(df, exg_dict)

    # =遍历每个策略，计算因子
    for strategy in stg_list:
        df, exg_dict = strategy.cal_factors(df, fin_df, fin_raw_df, exg_dict)

    # =添加不同的价格，方便使用不同的价格买入
    df['均价'] = df['成交额'] / df['成交量']
    for col in ['均价', '09:35收盘价', '09:45收盘价', '09:55收盘价']:
        df[f'{col}买入涨跌幅'] = df['收盘价'] / df[col] - 1
        df[f'下日_{col}买入涨跌幅'] = df[f'{col}买入涨跌幅'].shift(-1)
        exg_dict[f'下日_{col}买入涨跌幅'] = 'last'

    # =计算下个交易的相关情况
    df['下日_是否交易'] = df['是否交易'].shift(-1)
    df['下日_一字涨停'] = df['一字涨停'].shift(-1)
    df['下日_开盘涨停'] = df['开盘涨停'].shift(-1)
    df['下日_是否ST'] = df['股票名称'].str.contains('ST').shift(-1)
    df['下日_是否S'] = df['股票名称'].str.contains('S').shift(-1)
    df['下日_是否退市'] = df['股票名称'].str.contains('退').shift(-1)
    df['下日_开盘买入涨跌幅'] = df['开盘买入涨跌幅'].shift(-1)
    # 处理最后一根K线的数据
    state_cols = ['下日_是否交易', '下日_是否ST', '下日_是否S', '下日_是否退市']
    # df.loc[:, state_cols] = df.loc[:, state_cols].fillna(method='ffill')
    df[state_cols] = df[state_cols].fillna(method='ffill')
    # df.loc[:, ['下日_一字涨停', '下日_开盘涨停']] = df.loc[:, ['下日_一字涨停', '下日_开盘涨停']].fillna(value=False)
    df[['下日_一字涨停', '下日_开盘涨停']] = df[['下日_一字涨停', '下日_开盘涨停']].fillna(value=False)
    # _df = df.copy()

    # =将日线数据转化为对应period、offset的数据
    df = transfer_to_period_data(df, period_and_offset_df, period, exg_dict, each_offset)

    # =数据转换周期之后的操作
    for strategy in stg_list:
        df = strategy.after_resample(df)

    # =对数据进行整理
    # 删除上市的第一个周期
    if df.empty:
        print(f'{code}, 周期：{period} offset：{each_offset} 数据不足')
        return pd.DataFrame()
    df.drop([0], axis=0, inplace=True)  # 删除第一行数据

    # 删除2007年之前的数据，加入cut_day
    df = df[df['交易日期'] > max(pd.to_datetime('20061231'), cut_day)]
    # 计算下周期每天涨幅
    df['下周期每天涨跌幅'] = df['每天涨跌幅'].shift(-1)
    df['下周期涨跌幅'] = df['涨跌幅'].shift(-1)
    del df['每天涨跌幅']
    df = df[df['是否交易'] == 1]
    return df  # 返回计算好的数据


# ===并行处理每个股票的数据
if __name__ == '__main__':
    now = time.time()  # 用于记录运行时间
    # ===读取准备数据
    # 读取所有股票代码的列表
    stock_code_list = get_file_in_folder(stock_data_path, '.csv', filters=['bj'])
    print('股票数量：', len(stock_code_list))

    # 导入上证指数，保证指数数据和股票数据在同一天结束，不然会出现问题。
    index_data = import_index_data(index_path, back_trader_end=date_end)
    print(f'从指数获取最新交易日：{index_data["交易日期"].iloc[-1].strftime("%Y-%m-%d")}')

    # 导入「选股策略」文件夹中的所有选股策略
    stg_file = get_file_in_folder(root_path + '/program/选股策略/', '.py', filters=['_init_'], drop_type=True)
    # 遍历出「选股策略」文件夹中的每个策略的周期和offset
    period_dict = calc_period_offset(stg_file)
    # 举例：{'M': [0, -5], 'W': [0, 1, 2], '2W': [0, 1], 3: [0, 1, 2], 5: [0, 1, 2, 4]}

    # 加载所有周期所有offset的df
    period_and_offset_df = read_period_and_offset_file(period_offset_file)
    print(f'{"回测模式 所有offset都将运行" if backtest_mode else "实盘模式，仅对应offset将被运行"}')

    # ===遍历不同的周期进行选股
    for period in period_dict:  # 遍历所有选股策略中涉及的周期
        for each_offset in period_dict[period]:  # 遍历该周期所有涉及的offset
            if f'{period}_{each_offset}' not in period_and_offset_df.columns.to_list():  # 判断周期offset是否存在于预制csv中
                print(f'周期：{period} offset：{each_offset} 不在period_offset.csv中，更新csv后再试')
                continue
            if not backtest_mode:  # 判断回测or实盘模式
                # 实盘模式（非回测模式）
                is_current_period = is_in_current_period(period, each_offset, index_data, period_and_offset_df)
                if not is_current_period:  # 实盘模式不在正确周期则跳过
                    print(f'非回测模式，周期：{period} offset：{each_offset} 不计算')
                    continue
            print(f'正在计算周期：{period}，offset：{each_offset}')
            # ===导入每个策略，做好相应的准备
            stg_list = []  # 策略集合
            flow_fin_cols = []  # 流量型财务数据
            cross_fin_cols = []  # 横截面型财务数据
            add_fin_cols = []  # 最终需要的财务数据字段

            # 遍历每个策略
            for file in stg_file:
                cls = __import__('program.选股策略.%s' % file, fromlist=('',))
                # 判断是否有对应的period和offset
                if period != cls.period:
                    continue
                if not hasattr(cls, 'offset'):
                    print(f'策略{file}offset未定义，定义为[0]')
                    cls.offset = [0]
                if each_offset not in cls.offset:
                    continue
                stg_list.append(cls)
                # 如果该策略需要导入专属数据，就进行相关处理
                print('正在运行%s' % cls.name)
                cls.special_data()
                # 收集该策略所需要的财务数据
                flow_fin_cols += cls.flow_fin_cols
                cross_fin_cols += cls.cross_fin_cols
                add_fin_cols += cls.add_fin_cols

            # 如果没有当前计算周期的策略，直接跳过。
            if len(stg_list) == 0:
                continue

            # 为加快运算，根据subset_num和现有pkl文件情况对数据进行分割，这里算出分割日期。
            all_stock_data_path = root_path + f'/data/数据整理/all_stock_data_{period}_{each_offset}.pkl'
            if subset_num and subset_num > 0 and os.path.exists(all_stock_data_path):
                # subset_num被定义的情况下，subset_day用于切片算因子，cut_day用于算完后和all_stock_data拼接
                subset_day = index_data['交易日期'].iloc[-subset_num]
                cut_day = index_data['交易日期'].iloc[-int(subset_num * 0.3)]
                print(f'当前为加速计算模式，个股数据计算从{subset_day.strftime("%Y-%m-%d")}开始，'
                      f'合并数据从{cut_day.strftime("%Y-%m-%d")}开始，数据加速处理中')
            else:
                # subset_num为None或0 或原本all_stock_data pkl文件不存在，所有数据全算。
                # 全算的情况下，subset_day和cut_day就被定义为所有数据开始之前。
                subset_day = pd.to_datetime('1990-01-01')
                cut_day = pd.to_datetime('1990-01-01')

            # 财务字段数据去重
            flow_fin_cols = list(set(flow_fin_cols))
            cross_fin_cols = list(set(cross_fin_cols))
            add_fin_cols = list(set(add_fin_cols))

            # ===遍历每个股票，并行或者串行
            multiple_process = True  # True为并行，False为串行
            # 标记开始时间
            if multiple_process:
                df_list = Parallel(n_job)(delayed(calculate_by_stock)(code) for code in stock_code_list)
            else:
                df_list = []
                stock_code_list = stock_code_list
                for stock in stock_code_list:
                    res_df = calculate_by_stock(stock)
                    df_list.append(res_df)

            # ===最终整理数据
            # 合并为一个大的DataFrame
            all_stock_data = pd.concat(df_list, ignore_index=True)
            all_stock_data.sort_values(['交易日期', '股票代码'], inplace=True)  # ===将数据存入数据库之前，先排序、reset_index
            all_stock_data.reset_index(inplace=True, drop=True)
            # 将数据存储到pickle文件
            all_stock_data_path = root_path + f'/data/数据整理/all_stock_data_{period}_{each_offset}.pkl'
            if subset_num and subset_num > 0 and os.path.exists(all_stock_data_path):  # 判断是否需要合并旧文件
                # 载入旧文件
                old_all_stock_data = pd.read_pickle(all_stock_data_path)
                # 判断旧文件的日期是否能够被拼接
                if old_all_stock_data[old_all_stock_data['交易日期'] > cut_day].empty:
                    print(f'当前为加速计算模式，个股数据计算从{subset_day.strftime("%Y-%m-%d")}开始，'
                          f'合数据从{cut_day.strftime("%Y-%m-%d")}开始，all_stock_data结束日数据不够，'
                          f'建议subset_num置为None或调整到足够大后重跑本脚本。')
                    if backtest_mode:
                        # 如果为回测模式，直接卡住等待手工介入，实盘模式则不卡住。不想干预任意键可继续，不会让前面的东西都白算。
                        input('等待手工干预')
                old_all_stock_data = old_all_stock_data[old_all_stock_data['交易日期'] <= cut_day].copy()
                all_stock_data = pd.concat([old_all_stock_data, all_stock_data], ignore_index=True)

            # 保存本period offset的pkl文件
            all_stock_data.to_pickle(all_stock_data_path)
    print(f'耗时：{time.time() - now:.2f}秒')
