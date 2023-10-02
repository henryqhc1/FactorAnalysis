'''
2023分享会
author: 邢不行
微信: xbx9585
'''
from program.Functions import *
from program.Config import *

name = file_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')  # 当前文件的名字

period = 3  # 持仓周期（必填）

offset = auto_offset(period)

select_count = 3  # 选股数量（必填）

flow_fin_cols = ['R_np@xbx', 'R_revenue@xbx', 'R_op@xbx']  # 流量型财务字段

cross_fin_cols = ['B_total_equity_atoopc@xbx', 'B_total_liab@xbx', 'B_actual_received_capital@xbx', 'B_preferred_shares@xbx',
                  'B_total_assets@xbx', 'B_total_equity_atoopc@xbx', 'B_total_liab_and_owner_equity@xbx']  # 截面型财务字段

add_fin_cols = ['R_np@xbx_ttm', 'B_total_equity_atoopc@xbx', 'R_revenue@xbx_ttm', 'R_np@xbx_ttm同比', 'R_revenue@xbx_ttm同比',
                'R_np@xbx_单季同比', 'R_revenue@xbx_单季同比', 'B_total_liab@xbx', 'B_actual_received_capital@xbx', 'B_preferred_shares@xbx',
                'B_total_assets@xbx', 'B_total_equity_atoopc@xbx', 'B_total_liab_and_owner_equity@xbx', 'R_op@xbx_ttm']  # 最终需要加到数据上的财务字段


def special_data():
    '''
    处理策略需要的专属数据，非必要。
    :return:
    '''

    return


def before_merge_index(data, exg_dict, fill_0_list):
    '''
    合并指数数据之前的处理流程，非必要。
    :param data: 传入的数据
    :param exg_dict: resample规则
    :param fill_0_list: 合并指数时需要填充为0的数据
    :return:
    '''

    return data, exg_dict, fill_0_list


def merge_single_stock_file(data, exg_dict):
    '''
    合并策略需要的单个的数据，非必要。
    :param data:传入的数据
    :param exg_dict:resample规则
    :return:
    '''

    return data, exg_dict


def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    '''
    合并数据后计算策略需要的因子，非必要
    :param data:传入的数据
    :param fin_data:财报数据（去除废弃研报)
    :param fin_raw_data:财报数据（未去除废弃研报）
    :param exg_dict:resample规则
    :return:
    '''
    exg_dict['新版申万一级行业名称'] = 'last'
    exg_dict['总市值'] = 'last'

    # ===估值因子
    data[name + 'EP'] = data['R_np@xbx_ttm'] / data['总市值']  # 市盈率倒数
    data[name + 'BP'] = data['B_total_equity_atoopc@xbx'] / data['总市值']  # 市净率倒数
    data[name + 'SP'] = data['R_revenue@xbx_ttm'] / data['总市值']  # 市销率倒数
    exg_dict[name + 'EP'] = 'last'
    exg_dict[name + 'BP'] = 'last'
    exg_dict[name + 'SP'] = 'last'

    # ===动量因子
    data[name + 'Ret_252'] = data['收盘价'].shift(21) / data['收盘价'].shift(252) - 1
    exg_dict[name + 'Ret_252'] = 'last'

    # ===反转因子
    data[name + 'Ret_21'] = data['收盘价'] / data['收盘价'].shift(21) - 1
    exg_dict[name + 'Ret_21'] = 'last'

    # ===成长因子
    data[name + '净利润ttm同比'] = data['R_np@xbx_ttm同比']
    data[name + '营业收入ttm同比'] = data['R_revenue@xbx_ttm同比']
    data[name + '净利润单季同比'] = data['R_np@xbx_单季同比']
    data[name + '营业收入单季同比'] = data['R_revenue@xbx_单季同比']
    exg_dict[name + '净利润ttm同比'] = 'last'
    exg_dict[name + '营业收入ttm同比'] = 'last'
    exg_dict[name + '净利润单季同比'] = 'last'
    exg_dict[name + '营业收入单季同比'] = 'last'

    # ===杠杆因子
    data[name + 'MLEV'] = (data['总市值'] + data['B_total_liab@xbx']) / data['总市值']
    data[name + 'BLEV'] = (data[['B_actual_received_capital@xbx', 'B_preferred_shares@xbx']].sum(axis=1, skipna=True)) / data['总市值']
    data[name + 'DTOA'] = data['B_total_liab@xbx'] / data['B_total_assets@xbx']
    exg_dict[name + 'MLEV'] = 'last'
    exg_dict[name + 'BLEV'] = 'last'
    exg_dict[name + 'DTOA'] = 'last'

    # ===波动因子
    data[name + 'Std21'] = data['涨跌幅'].rolling(21).std()
    data[name + 'Std252'] = data['涨跌幅'].rolling(252).std()
    exg_dict[name + 'Std21'] = 'last'
    exg_dict[name + 'Std252'] = 'last'

    # ===流动性因子
    data[name + '换手率5'] = data['换手率'].rolling(5).mean()
    data[name + '换手率10'] = data['换手率'].rolling(10).mean()
    data[name + '换手率20'] = data['换手率'].rolling(20).mean()
    exg_dict[name + '换手率5'] = 'last'
    exg_dict[name + '换手率10'] = 'last'
    exg_dict[name + '换手率20'] = 'last'

    # ===盈利因子
    data[name + 'ROE'] = data['R_np@xbx_ttm'] / data['B_total_equity_atoopc@xbx']  # ROE 净资产收益率
    data[name + 'ROA'] = data['R_np@xbx_ttm'] / data['B_total_liab_and_owner_equity@xbx']  # ROA 资产收益率
    data[name + '净利润率'] = data['R_np@xbx_ttm'] / data['R_revenue@xbx_ttm']  # 净利润率：净利润 / 营业收入
    data[name + 'GP'] = data['R_op@xbx_ttm'] / data['B_total_assets@xbx']
    exg_dict[name + 'ROE'] = 'last'
    exg_dict[name + 'ROA'] = 'last'
    exg_dict[name + '净利润率'] = 'last'
    exg_dict[name + 'GP'] = 'last'

    # ===规模因子
    data[name + '总市值'] = np.log(data['总市值'] + 1)
    exg_dict[name + '总市值'] = 'last'

    return data, exg_dict


def after_resample(data):
    '''
    数据降采样之后的处理流程，非必要
    :param data: 传入的数据
    :return:
    '''
    return data


def filter_stock(all_data):
    '''
    过滤函数，在选股前过滤，必要
    :param all_data: 截面数据
    :return:
    '''
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

    all_data = cross_process(all_data)

    return all_data


def select_stock(all_data, count):
    '''
    选股函数，必要
    :param all_data: 截面数据
    :param count: 选股数量
    :return:
    '''
    df_for_group = all_data.copy()
    return all_data, df_for_group


def cross_process(df):
    '''
    截面处理数据
    df: 全部股票数据
    '''
    # ===估值
    df[name + 'EP排名'] = df.groupby('交易日期')[name + 'EP'].rank(ascending=True, method='min')
    df[name + 'BP排名'] = df.groupby('交易日期')[name + 'BP'].rank(ascending=True, method='min')
    df[name + 'SP排名'] = df.groupby('交易日期')[name + 'SP'].rank(ascending=True, method='min')
    df[name + '_估值'] = df[name + 'EP排名'] + df[name + 'BP排名'] + df[name + 'SP排名']
    df.drop(columns=[name + 'EP排名', name + 'BP排名', name + 'SP排名', name + 'EP', name + 'BP', name + 'SP'], inplace=True)

    # ===动量
    df[name + 'Ret_252排名'] = df.groupby('交易日期')[name + 'Ret_252'].rank(ascending=True, method='min')
    df[name + '_动量'] = df[name + 'Ret_252排名']
    df.drop(columns=[name + 'Ret_252排名', name + 'Ret_252'], inplace=True)

    # ===反转
    df[name + 'Ret_21排名'] = df.groupby('交易日期')[name + 'Ret_21'].rank(ascending=True, method='min')
    df[name + '_反转'] = df[name + 'Ret_21排名']
    df.drop(columns=[name + 'Ret_21排名', name + 'Ret_21'], inplace=True)

    # ===成长
    df[name + '净利润ttm同比排名'] = df.groupby('交易日期')[name + '净利润ttm同比'].rank(ascending=True, method='min')
    df[name + '营业收入ttm同比排名'] = df.groupby('交易日期')[name + '营业收入ttm同比'].rank(ascending=True, method='min')
    df[name + '净利润单季同比排名'] = df.groupby('交易日期')[name + '净利润单季同比'].rank(ascending=True, method='min')
    df[name + '营业收入单季同比排名'] = df.groupby('交易日期')[name + '营业收入单季同比'].rank(ascending=True, method='min')
    df[name + '_成长'] = df[name + '净利润ttm同比排名'] + df[name + '营业收入ttm同比排名'] + df[name + '净利润单季同比排名'] + df[name + '营业收入单季同比排名']
    df.drop(columns=[name + '净利润ttm同比排名', name + '营业收入ttm同比排名', name + '净利润单季同比排名', name + '营业收入单季同比排名',
                     name + '净利润ttm同比', name + '营业收入ttm同比', name + '净利润单季同比', name + '营业收入单季同比'], inplace=True)

    # ===杠杆
    df[name + 'MLEV排名'] = df.groupby('交易日期')[name + 'MLEV'].rank(ascending=True, method='min')
    df[name + 'BLEV排名'] = df.groupby('交易日期')[name + 'BLEV'].rank(ascending=True, method='min')
    df[name + 'DTOA排名'] = df.groupby('交易日期')[name + 'DTOA'].rank(ascending=True, method='min')
    df[name + '_杠杆'] = df[name + 'MLEV排名'] + df[name + 'BLEV排名'] + df[name + 'DTOA排名']
    df.drop(columns=[name + 'MLEV排名', name + 'BLEV排名', name + 'DTOA排名', name + 'MLEV', name + 'BLEV', name + 'DTOA'], inplace=True)

    # ===波动
    df[name + 'Std21排名'] = df.groupby('交易日期')[name + 'Std21'].rank(ascending=True, method='min')
    df[name + 'Std252排名'] = df.groupby('交易日期')[name + 'Std252'].rank(ascending=True, method='min')
    df[name + '_波动'] = df[name + 'Std21排名'] + df[name + 'Std252排名']
    df.drop(columns=[name + 'Std21排名', name + 'Std252排名', name + 'Std21', name + 'Std252'], inplace=True)

    # ===流动性
    df[name + '换手率5排名'] = df.groupby('交易日期')[name + '换手率5'].rank(ascending=True, method='min')
    df[name + '换手率10排名'] = df.groupby('交易日期')[name + '换手率10'].rank(ascending=True, method='min')
    df[name + '换手率20排名'] = df.groupby('交易日期')[name + '换手率20'].rank(ascending=True, method='min')
    df[name + '_流动性'] = df[name + '换手率5排名'] + df[name + '换手率10排名'] + df[name + '换手率20排名']
    df.drop(columns=[name + '换手率5排名', name + '换手率10排名', name + '换手率10排名', name + '换手率5', name + '换手率10', name + '换手率20'],
            inplace=True)

    # ===盈利
    df[name + 'ROE排名'] = df.groupby('交易日期')[name + 'ROE'].rank(ascending=True, method='min')
    df[name + 'ROA排名'] = df.groupby('交易日期')[name + 'ROA'].rank(ascending=True, method='min')
    df[name + '净利润率排名'] = df.groupby('交易日期')[name + '净利润率'].rank(ascending=True, method='min')
    df[name + 'GP排名'] = df.groupby('交易日期')[name + 'GP'].rank(ascending=True, method='min')
    df[name + '_盈利'] = df[name + 'ROE排名'] + df[name + 'ROA排名'] + df[name + '净利润率排名'] + df[name + 'GP排名']
    df.drop(columns=[name + 'ROE排名', name + 'ROA排名', name + '净利润率排名', name + 'GP排名',
                     name + 'ROE', name + 'ROA', name + '净利润率', name + 'GP'], inplace=True)

    # ===规模
    df[name + '总市值排名'] = df.groupby('交易日期')[name + '总市值'].rank(ascending=True, method='min')
    df[name + '_规模'] = df[name + '总市值排名']
    df.drop(columns=[name + '总市值排名', name + '总市值'], inplace=True)

    return df