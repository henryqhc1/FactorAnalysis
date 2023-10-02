"""
2023分享会
author: 邢不行
微信: xbx9585
选股策略框架
"""
import os.path
import pandas as pd
from Evaluate import *
from program.Config import *
from program.Functions import *
import warnings
import sys

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
# print输出中文表头对齐
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# =====判断是否是手工运行本程序
if_manual = True  # 默认手工运行
if len(sys.argv) > 1:  # 如果有外部输入，即为程序调用，不是手工运行程序
    strategy_file = sys.argv[1]  # 外出输入的参数，确认具体是跑哪个策略
    if_manual = False
else:
    # 合并不同offset及合并做图用
    equity_list = []
    fig_list = []

# =====动态导入选股策略
cls = __import__('program.选股策略.%s' % strategy_file, fromlist=('',))
print('\n策略文件：', strategy_file)

if not hasattr(cls, 'offset'):
    print('策略offset未定义，定义为[0]')
    cls.offset = [0]
print(f'周期：{cls.period} offset：{cls.offset}')

# 导入指数数据
index_data = import_index_data(index_path, back_trader_start=date_start, back_trader_end=date_end)
print(f'从指数获取最新交易日：{index_data["交易日期"].iloc[-1].strftime("%Y-%m-%d")}')

# 加载所有周期所有offset的df
period_and_offset_df = read_period_and_offset_file(period_offset_file)
print(f'{"回测模式 所有offset都将运行..." if backtest_mode else "实盘模式，仅对应offset将被运行"}')

for offset in cls.offset:  # 遍历所有offset
    is_current_period = is_in_current_period(cls.period, offset, index_data, period_and_offset_df)
    if not backtest_mode:  # 判断回测or实盘模式
        # 实盘模式
        if not is_current_period:
            # 不在正确周期
            print(f'非回测模式，最新交易日：{index_data["交易日期"].iloc[-1].strftime("%Y-%m-%d")} '
                  f'周期：{cls.period} offset：{offset} 不计算')
            continue
    print(f'\n======== 周期：{cls.period} offset：{offset} ========')

    # =====导入数据
    # 从pickle文件中读取整理好的所有股票数据
    df = pd.read_pickle(root_path + f'/data/数据整理/all_stock_data_{cls.period}_{offset}.pkl')
    if date_start:
        df = df[df['交易日期'] >= pd.to_datetime(date_start)]
    if date_end:
        df = df[df['交易日期'] <= pd.to_datetime(date_end)]

    # =====选股
    # ===过滤股票
    df = cls.filter_stock(df)

    # ===按照策略选股
    df, df_for_group = cls.select_stock(df, cls.select_count)
    # ===记录最近的的选股结果，并在原数据中删除
    # 最新选股
    new_select_stock = df[df['下周期每天涨跌幅'].isna() & (df['交易日期'] == df['交易日期'].max())].copy()
    # 删除数据
    df.dropna(subset=['下周期每天涨跌幅'], inplace=True)
    df_for_group.dropna(subset=['下周期每天涨跌幅'], inplace=True)

    # =====整理选中股票数据
    # ===按照开盘买入的方式，修正选中股票在下周期每天的涨跌幅。
    # 即将下周期每天的涨跌幅中第一天的涨跌幅，改成由开盘买入的涨跌幅
    df[f'下日_{buy_method}买入涨跌幅'] = df[f'下日_{buy_method}买入涨跌幅'].apply(lambda x: [x])
    df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(lambda x: x[1:])
    df['下周期每天涨跌幅'] = df[f'下日_{buy_method}买入涨跌幅'] + df['下周期每天涨跌幅']
    # 保存文件用于回测分析
    if os.path.exists(root_path + r'/data/分析目录/待分析/'):
        df.to_csv(root_path + f'/data/分析目录/待分析/{strategy_file}_{cls.period}_{offset}_{cls.select_count}.csv',
                  encoding='gbk', index=False)

    # ===挑选出选中股票
    df['股票代码'] += ' '
    df['股票名称'] += ' '
    group = df.groupby('交易日期')
    select_stock = pd.DataFrame()
    select_stock['股票数量'] = group['股票名称'].size()
    select_stock['买入股票代码'] = group['股票代码'].sum()
    select_stock['买入股票名称'] = group['股票名称'].sum()

    # =====计算资金曲线
    # 计算下周期每天的资金曲线
    select_stock['选股下周期每天资金曲线'] = group['下周期每天涨跌幅'].apply(
        lambda x: np.cumprod(np.array(list(x)) + 1, axis=1).mean(axis=0))

    # 扣除买入手续费
    select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'] * (1 - c_rate)  # 计算有不精准的地方
    # 扣除卖出手续费、印花税。最后一天的资金曲线值，扣除印花税、手续费
    select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'].apply(
        lambda x: list(x[:-1]) + [x[-1] * (1 - c_rate - t_rate)])

    # 计算下周期整体涨跌幅
    select_stock['选股下周期涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(lambda x: x[-1] - 1)
    # 计算下周期每天的涨跌幅
    select_stock['选股下周期每天涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(
        lambda x: list(pd.DataFrame([1] + x).pct_change()[0].iloc[1:]))
    del select_stock['选股下周期每天资金曲线']

    # 为了防止有的周期没有选出股票，创造一个空的df，用于填充不选股的周期
    empty_df = create_empty_data(index_data, cls.period, offset, period_and_offset_df)
    empty_df.update(select_stock)  # 将选股结果更新到empty_df上
    select_stock = empty_df

    # 计算整体资金曲线
    select_stock.reset_index(inplace=True)
    select_stock['资金曲线'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()

    # =====计算选中股票每天的资金曲线
    # 计算每日资金曲线
    equity = pd.merge(left=index_data, right=select_stock[['交易日期', '买入股票代码']], on=['交易日期'],
                      how='left', sort=True)  # 将选股结果和大盘指数合并

    equity['持有股票代码'] = equity['买入股票代码'].shift()
    if not new_select_stock.empty:
        equity = equity[equity['交易日期'] <= new_select_stock['交易日期'].max()]
    equity['持有股票代码'].fillna(method='ffill', inplace=True)
    equity.dropna(subset=['持有股票代码'], inplace=True)
    del equity['买入股票代码']
    equity['涨跌幅'] = select_stock['选股下周期每天涨跌幅'].sum()
    equity['equity_curve'] = (equity['涨跌幅'] + 1).cumprod()
    equity['benchmark'] = (equity['指数涨跌幅'] + 1).cumprod()

    # =====计算择时
    equity_notiming = pd.DataFrame()
    select_stock_notiming = pd.DataFrame()
    # 判断策略是否包含择时函数
    if hasattr(cls, "timing"):
        # 先备份一下旧的资金曲线 和 选股，后面作图之用
        equity_notiming = equity.copy()
        select_stock_notiming = select_stock.copy()

        # 把周期offset的df和equity进行合并，为了timing中可以确定调仓时间
        po_df = period_and_offset_df[['交易日期', f'{cls.period}_{offset}']].copy()
        po_df.rename(columns={f'{cls.period}_{offset}': '周期'}, inplace=True)
        po_df['周期'] = po_df['周期'].abs()
        equity = pd.merge(left=equity, right=po_df, on='交易日期', how='left')
        # 进行资金曲线择时
        signal_df = cls.timing(equity)  # 调用策略文件的中timing函数
        signal_df['signal'].fillna(method='ffill', inplace=True)  # 产生择时signal
        del equity['周期']
        # 把择时singal并入equity
        equity = pd.merge(left=equity, right=signal_df[['交易日期', 'signal']], on='交易日期', how='left')
        # 今天收盘的信号，明天才可以用
        equity['signal'] = equity['signal'].shift()
        equity['signal'].fillna(method='ffill', inplace=True)
        equity['signal'].fillna(value=1, inplace=True)

        # 根据signal重算资金曲线
        equity.loc[equity['signal'] == 0, '涨跌幅'] = 0
        equity.loc[equity['signal'] == 0, '持有股票代码'] = 'empty'
        equity['equity_curve'] = (equity['涨跌幅'] + 1).cumprod()

        # 根据signal 重算选股
        select_stock = pd.merge(left=select_stock, right=signal_df[['交易日期', 'signal']], on='交易日期', how='left')
        select_stock['signal'].fillna(method='ffill', inplace=True)
        select_stock['signal'].fillna(value=1, inplace=True)
        select_stock = select_stock[select_stock['signal'] == 1]
        select_stock.reset_index(inplace=True)

        signal = signal_df['signal'].iloc[-1]  # 存实盘数据用
    else:
        equity['signal'] = 1
        signal = 1

    # =====将选股策略的历史选股、最新选股，保存到本地文件。供后续使用
    # 保存最新的选股结果
    select_result_path = root_path + '/data/每日选股/选股策略/'
    if is_current_period:  # 仅在正确的周期offset情况下保存实盘文件
        save_select_result(select_result_path, new_select_stock, cls.name, cls.period, offset, cls.select_count, signal)
    else:
        # 不保存文件的话，也要读一下文件有没有，只要没有就创建（否则实盘配置的时候不方便）
        create_daily_result(select_result_path, cls.name, cls.period, offset, cls.select_count)

    # 保存历史选股结果，供后续轮动策略使用
    folder_path = root_path + '/data/回测结果/选股策略/'
    equity_to_csv(equity, strategy_file, cls.period, offset, cls.select_count, folder_path)

    # =====计算策略评价指标
    rtn, year_return, month_return = strategy_evaluate(equity, select_stock)
    # 如果有择时，需要拼一下未择时的东西
    if not equity_notiming.empty:
        rtn_notiming, year_return_notiming, month_return_notiming = strategy_evaluate(equity_notiming,
                                                                                      select_stock_notiming)
        rtn, year_return, month_return = merge_timing_data(rtn, rtn_notiming, year_return, year_return_notiming,
                                                           month_return, month_return_notiming)
        equity = pd.merge(left=equity, right=equity_notiming[['equity_curve']],
                          left_index=True, right_index=True, how='left', suffixes=('', '_notiming'))
    print(rtn, '\n', year_return)

    # =====画图、分组测试等
    if if_manual:

        # ===画图
        if not equity_notiming.empty:
            draw_data_dict = {'策略资金曲线': 'equity_curve_notiming', '策略资金曲线(带择时)': 'equity_curve',
                              '基准资金曲线': 'benchmark'}
            right_axis_dict = {'回撤(带择时)': 'dd2here'}
        else:
            draw_data_dict = {'策略资金曲线': 'equity_curve', '基准资金曲线': 'benchmark'}
            right_axis_dict = {'回撤': 'dd2here'}
        # 如果上面的函数不能画图，就用下面的画图
        fig = draw_equity_curve_plotly(equity,
                                       title=f'{strategy_file} 周期：{cls.period} 持股数量:{cls.select_count} offset:{offset} 换仓时间:{buy_method}',
                                       data_dict=draw_data_dict,
                                       right_axis=right_axis_dict,
                                       # date_col='交易日期',
                                       rtn_add=rtn)
        fig_list.append(fig)
        # 分组测试稳定性
        fig = robustness_test(df_for_group, bins=10, year_return_add=year_return)
        fig_list.append(fig)

        # 合并offset的资金曲线用
        equity_list.append(equity[equity.columns[equity.columns.to_series().apply(lambda x: 'equity_curve' in x)]])

if if_manual and backtest_mode:
    # 多offset进行合并
    if len(equity_list) > 1:
        print('\n===================合并所有offset=======================')
        # 多offset合成曲线
        equity, equity_notiming = merge_offset(equity_list, index_data)
        rtn, year_return, month_return = strategy_evaluate(equity, pd.DataFrame())
        # 如果有择时，需要拼一下未择时的东西
        if not equity_notiming.empty:
            rtn_notiming, year_return_notiming, month_return_notiming = strategy_evaluate(equity_notiming,
                                                                                          pd.DataFrame())
            rtn, year_return, month_return = merge_timing_data(rtn, rtn_notiming, year_return, year_return_notiming,
                                                               month_return, month_return_notiming)
            equity = pd.merge(left=equity, right=equity_notiming[['equity_curve']],
                              left_index=True, right_index=True, how='left', suffixes=('', '_notiming'))
        print(rtn, '\n', year_return)
        equity = equity.reset_index()

        # ===画图
        if not equity_notiming.empty:
            draw_data_dict = {'策略资金曲线': 'equity_curve_notiming', '策略资金曲线(带择时)': 'equity_curve',
                              '基准资金曲线': 'benchmark'}
            right_axis_dict = {'回撤(带择时)': 'dd2here'}
        else:
            draw_data_dict = {'策略资金曲线': 'equity_curve', '基准资金曲线': 'benchmark'}
            right_axis_dict = {'回撤': 'dd2here'}
        fig = draw_equity_curve_plotly(equity,
                                       title=f'{strategy_file} 周期：{cls.period} 持股数量:{cls.select_count} offset个数:{len(cls.offset)} 换仓时间:{buy_method}',
                                       data_dict=draw_data_dict,
                                       right_axis=right_axis_dict,
                                       date_col='交易日期', rtn_add=rtn,
                                       )
        fig_list = [fig] + fig_list

    # 储存并打开策略结果html
    merge_html(root_path, fig_list, strategy_file)
