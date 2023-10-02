'''
2023分享会
author: 邢不行
微信: xbx9585
'''
import pandas as pd

from Tools.utils.PlotFunctions import *
from program.Config import *
from Tools.utils.tFunctions import *
import warnings

warnings.filterwarnings('ignore')


# =====需要配置的东西=====
factor = 'Alpha95'  # 你想测试的因子
period = 3  # 配置读入数据的period
target = '下周期涨跌幅'  # 测试因子与下周期涨跌幅的IC，可以选择其他指标比如夏普率等
need_shift = False  # target这列需不需要shift，如果为True则将target这列向下移动一个周期
# =====需要配置的东西=====

# =====几乎不需要配置的东西=====
bins = 10  # 分箱数
limit = 100  # 1.某个周期至少有100只票，否则过滤掉这个周期；注意：limit需要大于bins；可能会造成不同因子开始时间不一致
next_ret = '下周期每天涨跌幅'  # 使用下周期每天涨跌幅画分组持仓走势图
offset = auto_offset(period)  # 根据配置的period获取offset
data_folder = root_path + f'/data/数据整理/'  # 配置读入数据的文件夹路径
industry_col = '新版申万一级行业名称'  # 配置行业的列名
# 行业名称更改信息，比如：21年之前的一级行业采掘在21年之后更名为煤炭
industry_name_change = {'采掘': '煤炭', '化工': '基础化工', '电气设备': '电力设备', '休闲服务': '社会服务', '纺织服装': '纺织服饰', '商业贸易': '商贸零售'}
market_value = '总市值'  # 配置总市值的列名
b_rate = 1.2 / 10000  # 买入手续费
s_rate = 1.12 / 1000  # 卖出手续费
factor_cls = __import__('program.选股策略.风格因子', fromlist=('',))  # 导入风格因子库
# =====几乎不需要配置的东西=====

# 读入数据
all_df = get_factor_by_period(data_folder, period, offset, target, need_shift, factor_cls)

# 如果target列向下shift1个周期，则更新下target指定的列
if need_shift:
    target = '下周期_' + target

# 删除必要字段为空的部分
df = all_df.dropna(subset=[factor, target, next_ret, market_value], how='any')
# ===根据limit，保留周期的股票数量大于limit的周期
# =拿到每个周期的股票数量
stock_nums = df.groupby('交易日期').size()
# =保留每个周期的股票数量大于limit的日期
save_dates = stock_nums[stock_nums > limit].index
# =如果交易日期save_dates中，则是否保留列为True
df['是否保留'] = df['交易日期'].map(lambda x: x in save_dates)
# =取出是否保留==True的数据
df = df[df['是否保留'] == True].reset_index(drop=True)

# 将数据按照交易日期和offset进行分组
df = offset_grouping(df, factor, bins)

# 生成一个包含图的列表，之后的代码每画出一个图都添加到该列表中，最后一起画出图
fig_list = []

# ===计算IC、累计IC以及IC的评价指标
corr, IC_info = IC_analysis(df, factor, target)
# =画IC走势图，并将IC图加入到fig_list中，最后一起画图
Rank_fig = draw_ic_plotly(x=corr['交易日期'], y1=corr['RankIC'], y2=corr['累计RankIC'], title='因子RankIC图', info=IC_info)
fig_list.append(Rank_fig)
# =画IC热力图（年份月份），并将图添加到fig_list中
# 处理IC数据，生成每月的平均IC
corr_month = get_corr_month(corr)
# 画图并添加
hot_fig = draw_hot_plotly(x=corr_month.columns, y=corr_month.index, z=corr_month, title='RankIC热力图(行：年份，列：月份)')
fig_list.append(hot_fig)

# ===计算分组资金曲线、分箱图、分组持仓走势
group_curve, group_value, group_hold_value = group_analysis(df, next_ret, b_rate, s_rate)
# =画分组资金曲线...
cols_list = [col for col in group_curve.columns if '第' in col]
group_fig = draw_line_plotly(x=group_curve['交易日期'], y1=group_curve[cols_list], y2=group_curve['多空净值'], if_log=True, title='分组资金曲线')
fig_list.append(group_fig)
# =画分箱净值图
group_fig = draw_bar_plotly(x=group_value['分组'], y=group_value['净值'], title='分组净值')
fig_list.append(group_fig)
# =画分组持仓走势
group_fig = draw_line_plotly(x=group_hold_value['时间'], y1=group_hold_value[cols_list], update_xticks=True, if_log=False, title='分组持仓走势')
fig_list.append(group_fig)

# ===计算风格暴露
style_corr = style_analysis(df, factor)
if not style_corr.empty:
    # =画风格暴露图
    style_fig = draw_bar_plotly(x=style_corr['风格'], y=style_corr['相关系数'], title='因子风格暴露图')
    fig_list.append(style_fig)

# ===行业
# =计算行业平均IC以及行业占比
industry_data = industry_analysis(df, factor, target, industry_col, industry_name_change)
# =画行业分组RankIC
industry_fig1 = draw_bar_plotly(x=industry_data[industry_col], y=industry_data['RankIC'], title='行业RankIC图')
fig_list.append(industry_fig1)
# =画行业暴露
industry_fig2 = draw_double_bar_plotly(x=industry_data[industry_col], y1=industry_data['行业占比_第一组'], y2=industry_data['行业占比_最后一组'], title='行业占比（可能会受到行业股票数量的影响）')
fig_list.append(industry_fig2)

# ===市值
# =计算不同市值分组内的平均IC以及市值占比
market_value_data = market_value_analysis(df, factor, target, market_value, bins)
# =画市值分组RankIC
market_value_fig1 = draw_bar_plotly(x=market_value_data['市值分组'], y=market_value_data['RankIC'], title='市值分组RankIC')
fig_list.append(market_value_fig1)
# =画市值暴露
market_value_fig2 = draw_double_bar_plotly(x=market_value_data['市值分组'], y1=market_value_data['市值占比_第一组'], y2=market_value_data['市值占比_最后一组'], title='市值占比')
fig_list.append(market_value_fig2)

# ===整合上面所有的图
merge_html(root_path, fig_list=fig_list, strategy_file=f'{factor}因子分析报告')
