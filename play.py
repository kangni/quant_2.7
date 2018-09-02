#%%
from collections import namedtuple
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import scipy.stats as scs
import seaborn as sb
import abupy
from abupy import ABuMarketDrawing
from abupy import ABuSymbolPd
from abupy import ABuIndustries
from abupy import ABuScalerUtil

#%%
linspce_exp = np.linspace(0, 7, 10)
linspce_exp
stock_cnt = 200
view_days = 504
stock_day_change = np.random.standard_normal((stock_cnt, view_days))
stock_example_a = stock_day_change[:2, :5]
print(stock_example_a)
print('====================')
print('int data')
print(stock_example_a.astype(int))
print('====================')
print('float data')
print(np.around(stock_example_a, 2))
print('====================')
print('np.nan')
tmp_test = stock_example_a.copy()
tmp_test[0][0] = np.nan
print(tmp_test)
print('====================')
print('np.nan_to_num')
tmp_test = np.nan_to_num(tmp_test)
print(tmp_test)
print('====================')
print('bool')
mask = stock_example_a > 0.5
print(mask)

stock_example_b = stock_day_change[-2:, -5:]
print('original data')
print(stock_example_a)
print(stock_example_b)
print('====================')
print(np.maximum(stock_example_a, stock_example_b))

print(stock_example_a)
print('====================')
print(np.diff(stock_example_a))
print(np.diff(stock_example_a, axis=0))

tmp_test = stock_example_b
print(tmp_test)
print(np.where(tmp_test > 0.5, 1, 0))
print(np.where(np.logical_and(tmp_test > 0.5, tmp_test < 1), 1, 0))
print(np.where(np.logical_or(tmp_test > 0.5, tmp_test < -0.5), 1, 0))

#%%
np.save('stock_day_change', stock_day_change)
stock_day_change = np.load('stock_day_change.npy')
stock_day_change.shape

#%%
stock_day_change_four = stock_day_change[:4, :4]
print(stock_day_change_four)
print('====================')
print('4只股票在4天内的表现')
print('最大涨幅 {}'.format(np.max(stock_day_change_four, axis=1)))
print('最大跌幅 {}'.format(np.min(stock_day_change_four, axis=1)))
print('振幅幅度 {}'.format(np.std(stock_day_change_four, axis=1)))
print('平均涨跌 {}'.format(np.mean(stock_day_change_four, axis=1)))
print('====================')
print('某一交易日4只股票的表现')
print('最大涨幅 {}'.format(np.max(stock_day_change_four, axis=0)))
print('最大涨幅股票 {}'.format(np.argmax(stock_day_change_four, axis=0)))

#%%
a_investor = np.random.normal(loc=100, scale=50, size=(10000, 1))
b_investor = np.random.normal(loc=100, scale=20, size=(10000, 1))
a_mean = a_investor.mean()
a_std = a_investor.std()
a_var = a_investor.var()
b_mean = b_investor.mean()
b_std = b_investor.std()
b_var = b_investor.var()

print('a 交易者期望 {0:.2f} 元，标准差 {1:.2f}，方差 {2:.2f}'.format(a_mean, a_std, a_var))
print('b 交易者期望 {0:.2f} 元，标准差 {1:.2f}，方差 {2:.2f}'.format(b_mean, b_std, b_var))

#%%
plt.plot(a_investor)
plt.axhline(a_mean + a_std, color='r')
plt.axhline(a_mean, color='y')
plt.axhline(a_mean -  a_std, color='g')
print('a investor')

plt.plot(b_investor)
plt.axhline(b_mean + b_std, color='r')
plt.axhline(b_mean, color='y')
plt.axhline(b_mean - b_std, color='g')
print('b investor')

#%%
# pdf()：在统计学中称为概率密度函数，是指在某个确定的取值点附近的可能性的函数，
# 将概率值分配给各个事件，得到事件的概率分布，让事件数值化。
first_stock = stock_day_change[0]
stock_mean = first_stock.mean()
stock_std = first_stock.std()
print('股票 0 mean 均值期望:{:.3f}'.format(stock_mean))
print('股票 0 std 振幅标准差:{:.3f}'.format(stock_std))
plt.hist(first_stock, bins=50, normed=True)
fit_linspece = np.linspace(first_stock.min(), first_stock.max())
pdf = scs.norm(stock_mean, stock_std).pdf(fit_linspece)
plt.plot(fit_linspece, pdf, lw=2, c='r')

#%%
keep_days = 50
stock_day_change_test = stock_day_change[:stock_cnt, :view_days-keep_days]
stock_lower_array = np.argsort(np.sum(stock_day_change_test, axis=1))[:3]
print('前 454 天中跌幅最大的三只股票跌幅：{}'.format(np.sort(np.sum(stock_day_change_test, axis=1))[:3]))
print('前 454 天中跌幅最大的三只股票序号: {}'.format(stock_lower_array))

#%%
def show_buy_lower(stock_ind):
    '''
    :param stock_ind: 股票序号，即在 stock_day_change 中的位置
    :return:
    '''
    # stock_cnt = 200
    # view_days = 504
    # stock_day_change = np.random.standard_normal((stock_cnt, view_days))
    stock_day_change_test = stock_day_change[:stock_cnt, :view_days-keep_days]
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
    axs[0].plot(np.arange(0, view_days - keep_days), stock_day_change_test[stock_ind].cumsum())
    cs_buy = stock_day_change[stock_ind][view_days - keep_days:view_days].cumsum()
    axs[1].plot(np.arange(view_days - keep_days, view_days), cs_buy)
    return cs_buy[-1]

#%%
profit = 0
for stock_ind in stock_lower_array:
    profit += show_buy_lower(stock_ind)
print('买入第 {} 只股票，从第 454 个交易日开始持有盈亏：{:.2f}%'.format(stock_lower_array, profit))

#%%
gamblers = 100
def casino(win_rate, win_once=1, loss_once=1, commission=0.01):
    """
    win_rate: 输赢的概率
    win_once: 每次赢的钱数
    loss_once: 每次输的钱数
    commission: 手续费 1%
    """
    my_money = 1000000
    play_cnt = 10000
    commission = commission
    for _ in np.arange(0, play_cnt):
        w = np.random.binomial(1, win_rate)
        if w:
            my_money += win_once
        else:
            my_money -= loss_once
        my_money -= commission
        if my_money <= 0:
            break
    return my_money

#%%
heaven_moneys = [casino(0.5, commission=0) for _ in np.arange(0, gamblers)]
plt.hist(heaven_moneys, bins=30)

cheat_moneys = [casino(0.4, commission=0) for _ in np.arange(0, gamblers)]
plt.hist(cheat_moneys, bins=30)

commission_moneys = [casino(0.5, commission=0.01) for _ in np.arange(0, gamblers)]
plt.hist(commission_moneys, bins=30)

moneys = [casino(0.5, commission=0.01, win_once=1.02, loss_once=0.98) for _ in np.arange(0, gamblers)]
plt.hist(moneys, bins=30)

moneys = [casino(0.45, commission=0.01, win_once=1.02, loss_once=0.98) for _ in np.arange(0, gamblers)]
plt.hist(moneys, bins=30)

#%%
stock_day_change = np.load('stock_day_change.npy')
pd.DataFrame(stock_day_change)[:5]

#%%
stock_symbols = ['stock' + str(x) for x in range(stock_day_change.shape[0])]
pd.DataFrame(stock_day_change, index=stock_symbols)[:2]
days = pd.date_range('2017-1-1', periods=stock_day_change.shape[1], freq='1d')
stock_symbols = ['stock' + str(x) for x in range(stock_day_change.shape[0])]
df = pd.DataFrame(stock_day_change, index=stock_symbols, columns=days)
df = df.T
df.head()
df_20 = df.resample('21D', how='mean')
df_20.head()

#%%
df_stock0 = df['stock0']
df_stock0.head()
df_stock0.cumsum().plot()

#%%
df_stock0_5 = df_stock0.cumsum().resample('5D').ohlc()
df_stock0_20 = df_stock0.cumsum().resample('21D').ohlc()
df_stock0_5.head()

#%%
ABuMarketDrawing.plot_candle_stick(df_stock0_5.index,
                                   df_stock0_5['open'].values,
                                   df_stock0_5['high'].values,
                                   df_stock0_5['low'].values,
                                   df_stock0_5['close'].values,
                                   np.random.random(len(df_stock0_5)),
                                   None, 'stock', day_sum=False,
                                   html_bk=False, save=False)

#%%
print(type(df_stock0_5['open'].values))
print(df_stock0_5['open'].index)
print(df_stock0_5.columns)

#%%
tsla_df = ABuSymbolPd.make_kl_df('usTSLA', n_folds=3)
tsla_df.tail()
tsla_df[['close', 'volume']].plot(subplots=True, style=['r', 'g'], grid=True)

#%%
# pandas 的 DataFrame 对象总览数据的
# 函数 info() 的用途是查看数据是否有缺失
# 以及各个子数据的数据类型
tsla_df.info()
# describe() 的用途是分别展示每组数据的统计信息
tsla_df.describe()

#%%
tsla_df.loc['2018-01-01':'20180810']
tsla_df.loc['2018-01-01':'20180810', 'open']
tsla_df.iloc[0:5, 2:6]
tsla_df.iloc[:, 2:6]
tsla_df.iloc[:5]
tsla_df.close[0:3]
tsla_df[['close', 'high', 'low']][0:3]

#%%
# "涨跌幅大于 8%" & "交易成交量大于统计周期内的平均值的 2.5 倍"
tsla_df[np.abs(tsla_df.p_change) > 8]
tsla_df.volume > 2.5 * tsla_df.volume.mean()
tsla_df[(np.abs(tsla_df.p_change) > 8) & (tsla_df.volume > 2.5 * tsla_df.volume.mean())]
tsla_df.sort_index(by='p_change')[:5]
tsla_df.sort_index(by='p_change', ascending=False)[:5]

#%%
# 缺失数据处理
#
# 一行中的数据存在 na 就删除这行
tsla_df.dropna()
# 如果一行的数据全为 na 就删除这行
tsla_df.dropna(how='all')
# 使用指定值填充 na，inplace 代表就地操作，即不返回新的序列在原始序列上修改
tsla_df.fillna(tsla_df.mean(), inplace=False)

#%%
# pct_change()：对序列从第二项开始向前做减法后再除以前一项。
# 因为此函数针对价格序列的操作结果即是涨跌幅序列，在股票量化等领域经常使用。
tsla_df.close[-3:]
tsla_df.close.pct_change()[-3:]
change_ratio = tsla_df.close.pct_change()
change_ratio.tail()

# round() 函数使用
# 将 change_ratio 转变成 tsla_df.netChangeRatio 字段一样的百分数
# 同样保留两位小数
np.round(change_ratio[-5:] * 100, 2)

# 使用 Series 对象的 map() 函数针对列数据 atr21，实现和上面例子 round() 一样的功能
format = lambda x: '%.2f' % x
tsla_df.atr21.map(format).tail()

(change_ratio[-5:]*100).map(format)

#%%
# 数据本地序列化操作
#
# 使用 to_csv 保存 DataFrame 对象，columns 列名称
tsla_df.to_csv('tsla_df.csv', columns=tsla_df.columns, index=True)
tsla_df_load = pd.read_csv('tsla_df.csv', parse_dates=True, index_col=0)
tsla_df_load.tail()

#%%
# 如果把涨跌数据分类成十份，TOP 10% 振幅的就被认为是异常表现的振幅，\
# 我们的需求是鉴定 TSLA 的异常振幅阀值是多少。
tsla_df.p_change.hist(bins=80)

#%%
# qcut() 函数将涨跌幅数据进行平均分类。
# value_counts() 函数经常和 qcut() 函数一起使用，便于更直观地显示分离结果
# 需要注意的是，只有 Series 对象才有 value_counts() 方法。
cats = pd.qcut(np.abs(tsla_df.p_change), 10)
cats.value_counts()

#%%
# 数据的离散化
# 如果有自己的分类规则，应该使用 pd.cut() 并传入 bins
# pd.cut() 函数经常会和 pd.get_dummies() 函数配合使用，将数据由
# 连续数值类型变成离散类型，即数据的离散化。
# get_dummies() 生成离散化的哑变量矩阵多用于 ML 中监督学习问题的分类，
# 使用它来作为训练数据使用。
bins = [-np.inf, -8.0, -6, -4, -2, 0, 2, 4, 6, 8, np.inf]
cats = pd.cut(tsla_df.p_change, bins)
cats.value_counts()

change_ration_dummies = pd.get_dummies(cats, prefix='cr_dummies')
change_ration_dummies.tail()

# 与 tsla_df 表进行合并
pd.concat([tsla_df, change_ration_dummies], axis=1).tail()

pd.concat([tsla_df[tsla_df.p_change > 10],
    tsla_df[tsla_df.atr14 > 16]], axis=0)
tsla_df[tsla_df.p_change > 10].append(tsla_df[tsla_df.atr14 > 16])

stock_a = pd.DataFrame({'stock_a': ['a', 'b', 'c', 'd', 'a'],
    'data': range(5)})
stock_b = pd.DataFrame({'stock_b': ['a', 'b', 'c'],
    'data': range(3)})
pd.merge(stock_a, stock_b, left_on='stock_a', right_on='stock_b')

#%%
tsla_df['positive'] = np.where(tsla_df.p_change > 0, 1, 0)
tsla_df.tail()
# 使用 pd.crosstab() 构建交叉表
xt = pd.crosstab(tsla_df.date_week, tsla_df.positive)
xt

# 下面是经常和 pd.crosstab() 配套出现的代码，可以说是一个套路
# 目的是求出所占比例
xt_pct = xt.div(xt.sum(1).astype(float), axis=0)
xt_pct

xt_pct.plot(
    figsize=(8, 5),
    kind='bar',
    stacked=True,
    title='date_week -> positive'
)
plt.xlabel('date_week')
plt.ylabel('positive')

#%%
# 创建透视表
tsla_df.pivot_table(['positive'], index=['date_week'])
tsla_df.groupby(['date_week', 'positive'])['positive'].count()

#%%
# 下面根據「跳空缺口」的理論來尋找 TSLA 的跳空缺口，
# 但並不是使用傳統的跳空定義方式，因為這種方式很容易夾雜普通缺口。
# 定義缺口的方式如下：
# 1. 今天如果是上漲趨勢，那麼跳空的確定需要今天的最低價格大於昨天收盤價格一個閥值以上，確定向上跳空；
# 2. 今天如果是下跌趨勢，那麼跳空的確定需要昨天的收盤價格大於今天最高價格一個閥值以上，確定向下跳空。

jump_threshold = tsla_df.close.median() * 0.03
jump_threshold

#%%
def judge_jump(today):
    global jump_pd
    if today.p_change > 0 and \
        (today.low - today.pre_close) > jump_threshold:
        today['jump'] = 1
        today['jump_power'] = (today.low - today.pre_close) / jump_threshold
        jump_pd = jump_pd.append(today)
    elif today.p_change < 0 and \
        (today.pre_close - today.high) > jump_threshold:
        today['jump'] = -1
        today['jump_power'] = (today.pre_close - today.high) / jump_threshold
        jump_pd = jump_pd.append(today)

jump_pd = pd.DataFrame()
# for kl_index in np.arange(0, tsla_df.shape[0]):
    # today = tsla_df.ix[kl_index]
    # judge_jump(today)

tsla_df.apply(judge_jump, axis=1)
jump_pd.filter(['jump', 'jump_power', 'close', 'date', 'p_change', 'pre_close'])


#%%
ABuMarketDrawing.plot_candle_form_klpd(tsla_df, view_indexs=jump_pd.index)

#%%
r_symbol = 'usQCOM'
# 获取和 TSLA 处于同一行业的股票
p_date, _ = ABuIndustries.get_industries_panel_from_target(r_symbol, show=False)
type(p_date)
# 三维数据
p_date
# 其中一个切面数据
p_date['usNOK'].tail()

#%%
# 高维的 Panel 通过轴向的互换等空间操作，可以高效灵活地变换出各种数据形式。
p_data_it = p_date.swapaxes('items', 'minor')
p_data_it

# 通过拿出 Items axis 中的 close 来选取所有股票的 close，形成一个新的横切面数据
p_data_it_close = p_data_it['close'].dropna(axis=0)
p_data_it_close.tail()

#%%
p_data_it_close = ABuScalerUtil.scaler_std(p_data_it_close)
p_data_it_close.plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Price')
plt.xlabel('Time')

#%%
tsla_df = ABuSymbolPd.make_kl_df('usTSLA', n_folds=2)
tsla_df.tail()

#%%
def plot_demo(axs=None, just_series=False):
    """
    :param axs: axs为子画布
    :param just_series: 是否只绘制一条收盘曲线使用 Series
    """
    drawer = plt if axs is None else axs
    drawer.plot(tsla_df.close, c='r')
    if not just_series:
        drawer.plot(tsla_df.close.index, tsla_df.close.values + 10, c='g')
        drawer.plot(tsla_df.close.index.tolist(), (tsla_df.close.values + 20).tolist(), c='b')
    plt.xlabel('time')
    plt.ylabel('close')
    plt.title('TSLA CLOSE')
    plt.grid(True)

plot_demo()

#%%
_, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
drawer = axs[0][0]
plot_demo(drawer)
drawer.legend(['Series', 'Numpy', 'List'], loc=0)

drawer = axs[0][1]
plot_demo(drawer)
drawer.legend(['Series', 'Numpy', 'List'], loc=1)

drawer = axs[1][0]
plot_demo(drawer)
drawer.legend(['Series', 'Numpy', 'List'], loc=2)

drawer = axs[1][1]
plot_demo(drawer)
drawer.legend(['Series', 'Numpy', 'List'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#%%
__colorup__ = 'green'
__colordown__ = 'red'
tsla_part_df = tsla_df[:90]
fig, ax = plt.subplots(figsize=(14, 7))
qutotes = []
for index, (d, o, c, h, l) in enumerate(
    zip(tsla_part_df.index, tsla_part_df.open, tsla_part_df.close, tsla_part_df.high, tsla_part_df.low)):
    d = mpf.date2num(d)
    val = (d, o, c, h, l)
    qutotes.append(val)
mpf.candlestick_ochl(ax, qutotes, width=0.6, colorup=__colorup__, colordown=__colordown__)
ax.autoscale_view()
ax.xaxis_date()

#%%
# 使用封装后的交互可视化 Bokeh 库
ABuMarketDrawing.plot_candle_form_klpd(tsla_df, html_bk=True)

#%%
demo_list = np.array([2, 4, 16, 20])
demo_window = 3
pd.rolling_std(demo_list, window=demo_window, center=False) * np.sqrt(demo_window)
pd.Series([2, 4, 16]).std() * np.sqrt(demo_window)
pd.Series([4, 16, 20]).std() * np.sqrt(demo_window)
np.sqrt(pd.Series([2, 4, 16]).var() * demo_window)

#%%
tsla_df_copy = tsla_df.copy()
tsla_df_copy['return'] = np.log(tsla_df['close'] / tsla_df['close'].shift(1))
tsla_df_copy['mov_std'] = pd.rolling_std(tsla_df_copy['return'], window=20, center=False) * np.sqrt(20)
tsla_df_copy['std_ewm'] = pd.ewmstd(tsla_df_copy['return'], span=20, min_periods=20, adjust=True) * np.sqrt(20)
tsla_df_copy[['close', 'mov_std', 'std_ewm', 'return']].plot(subplots=True, grid=True)

#%%
tsla_df.close.plot()
pd.rolling_mean(tsla_df.close, window=30).plot()
pd.rolling_mean(tsla_df.close, window=60).plot()
pd.rolling_mean(tsla_df.close, window=90).plot()
plt.legend(['close', '30 mv', '60 mv', '90 mv'], loc='best')

#%%
low_to_high_df = tsla_df.iloc[tsla_df[(tsla_df.close > tsla_df.open) & (tsla_df.key != tsla_df.shape[0] - 1)].key.values + 1]
change_ceil_floor = np.where(low_to_high_df['p_change'] > 0, np.ceil(low_to_high_df['p_change']), np.floor(low_to_high_df['p_change']))
change_ceil_floor = pd.Series(change_ceil_floor)
print('低开高收的下一个交易日所有下跌的跌幅取整和sum：' + str(change_ceil_floor[change_ceil_floor < 0].sum()))
print('低开高收的下一个交易日所有上涨的跌幅取整和sum：' + str(change_ceil_floor[change_ceil_floor > 0].sum()))
_, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
change_ceil_floor.value_counts().plot(kind='bar', ax=axs[0][0])
change_ceil_floor.value_counts().plot(kind='barh', ax=axs[0][1])
change_ceil_floor.value_counts().plot(kind='kde', ax=axs[1][0])
change_ceil_floor.value_counts().plot(kind='pie', ax=axs[1][1])

#%%
# use Seaborn lib
sb.distplot(tsla_df['p_change'], bins=80)
sb.boxplot(x='date_week', y='p_change', data=tsla_df)
sb.jointplot(tsla_df['high'], tsla_df['low'])

#%%
change_df = pd.DataFrame({'tsla': tsla_df.p_change})
change_df = change_df.join(pd.DataFrame({'goog': ABuSymbolPd.make_kl_df('usGOOG', n_folds=2).p_change}), how='outer')
change_df = change_df.join(pd.DataFrame({'aapl': ABuSymbolPd.make_kl_df('usAAPL', n_folds=2).p_change}), how='outer')
change_df = change_df.join(pd.DataFrame({'fb': ABuSymbolPd.make_kl_df('usFB', n_folds=2).p_change}), how='outer')
change_df = change_df.join(pd.DataFrame({'bidu': ABuSymbolPd.make_kl_df('usBIDU', n_folds=2).p_change}), how='outer')
change_df = change_df.dropna()
change_df.tail()

# 使用 corr 计算数据的相关性
# 数据可视化的目的是通过可视化更直观深入地理解数据，发现数据之间的关系，进一步指导策略，发现问题。
corr = change_df.corr()
_, ax = plt.subplots(figsize=(8, 5))
sb.heatmap(corr, ax=ax)

#%%
# 标注策略交易区间
# 假定我们运行了一个量化策略，执行回测，其中一个操作是：2017-07-28 买入股票 TSLA，2017-09-05 卖出。我们的需求就是 \
# 在收盘价格时间序列的基础上标明上面这个持有区间。
def plot_trade(buy_date, sell_date):
    start = tsla_df[tsla_df.index == buy_date].key.values[0]
    end = tsla_df[tsla_df.index == sell_date].key.values[0]
    plot_demo(just_series=True)
    plt.fill_between(tsla_df.index, 0, tsla_df['close'], color='blue', alpha=.08)
    if tsla_df['close'][end] > tsla_df['close'][start]:
        plt.fill_between(tsla_df.index[start:end], 0, tsla_df['close'][start:end], color='green', alpha=.38)
        is_win = True
    else:
        plt.fill_between(tsla_df.index[start:end], 0, tsla_df['close'][start:end], color='red', alpha=.38)
        is_win = False
    plt.ylim(np.min(tsla_df['close']) - 5, np.max(tsla_df['close']) + 5)
    plt.legend(['close'], loc='best')
    return is_win

plot_trade('2017-07-28', '2017-09-05')

#%%
# 标明策略卖出原因
# 假定我们运行了一个量化策略，执行回测，我们在 2017-07-28 买入股票是因为量化策略发出信号，TSLA 满足了买入条件，所以 \
# 买入了股票，而选择在 2017-09-05 卖出的原因却可能有很多种可能。现在假定卖出的原因只有止盈和止损两种，我们的需求就是标明原因。
def plot_trade_with_annotate(buy_date, sell_date):
    is_win = plot_trade(buy_date, sell_date)
    plt.annotate('sell for stop win' if is_win else 'sell for stop loss', xy=(sell_date, tsla_df['close'].asof(sell_date)),
        arrowprops=dict(facecolor='yellow'),
        horizontalalignment='left', verticalalignment='top')

plot_trade_with_annotate('2017-08-01', '2017-10-02')
plot_trade_with_annotate('2018-05-21', '2018-07-30')
plot_trade_with_annotate('2018-08-02', '2018-08-17')

#%%
# 将多只股票的价格在同一段统计周期内可视化，通过可视化发现股票间的走势关系和相关性等特征。
goog_df = ABuSymbolPd.make_kl_df('usGOOG', n_folds=2)
print(round(goog_df.close.mean(), 2), round(goog_df.close.median(), 2))
goog_df.tail()

def plot_two_stock(tsla, goog, axs=None):
    drawer = plt if axs is None else axs
    drawer.plot(tsla, c='r')
    drawer.plot(goog, c='g')
    drawer.grid(True)
    drawer.legend(['tsla', 'google'], loc='best')

plot_two_stock(tsla_df.close, goog_df.close)
plt.title('TSLA and Google CLOSE')
plt.xlabel('time')
plt.ylabel('close')

#%%
def two_mean_list(one, two, type_look='look_max'):
    one_mean = one.mean()
    two_mean = two.mean()
    if type_look == 'look_max':
        one, two = (one, one_mean / two_mean * two) if one_mean > two_mean else (one * two_mean / one_mean, two)
    elif type_look == 'look min':
        one, two = (one * two_mean / one_mean, two ) if one_mean > two_mean else (one, two * one_mean / two_mean)
    return one, two

def regular_std(group):
    """
    z-score 规范化
    """
    return (group - group.mean()) / group.std()

def regular_mm(group):
    """
    最小-最大规范化
    """
    return (group - group.min()) / (group.max() - group.min())

#%%
_, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
drawer = axs[0][0]
plot_two_stock(regular_std(tsla_df.close), regular_std(goog_df.close), drawer)
drawer.set_title('(group - group.mean()) / group.std()')

drawer = axs[0][1]
plot_two_stock(regular_mm(tsla_df.close), regular_mm(goog_df.close), drawer)
drawer.set_title('(group - group.min()) / (group.max() - group.min())')

drawer = axs[1][0]
one, two = two_mean_list(tsla_df.close, goog_df.close, type_look='look_max')
plot_two_stock(one, two, drawer)
drawer.set_title('two_mean_list type_look=look_max')

drawer = axs[1][1]
one, two = two_mean_list(tsla_df.close, goog_df.close, type_look='look_min')
plot_two_stock(one, two, drawer)
drawer.set_title('two_mean_list type_look=look_min')

#%%
_, ax1 = plt.subplots()
ax1.plot(tsla_df.close, c='r', label='tsla')
ax1.legend(loc=2)
ax1.grid(False)

ax2 = ax1.twinx()
ax2.plot(goog_df.close, c='g', label='goog')
ax2.legend(loc=1)

#%%
cs_max = tsla_df.close.max()
cs_min = tsla_df.close.min()
sp382 = (cs_max - cs_min) * 0.382 + cs_min
sp618 = (cs_max - cs_min) * 0.618 + cs_min
print('视觉上的 382: ' + str(round(sp382, 2)))
print('视觉上的 618: ' + str(round(sp618, 2)))

demo_list = [1, 1, 1, 100, 100, 100, 100, 100, 100, 100]
pd.Series(demo_list).sort_values(inplace=False)
scs.scoreatpercentile(demo_list, 38.2)

sp382_stats = scs.scoreatpercentile(tsla_df.close, 38.2)
sp618_stats = scs.scoreatpercentile(tsla_df.close, 61.8)
print('统计上的 sp382 : ' + str(round(sp382_stats, 2)))
print('统计上的 sp618 : ' + str(round(sp618_stats, 2)))

#%%
def plot_golden():
    above618 = np.maximum(sp618, sp618_stats)
    below618 = np.minimum(sp618, sp618_stats)
    above382 = np.maximum(sp382, sp382_stats)
    below382 = np.minimum(sp382, sp382_stats)

    plt.plot(tsla_df.close)
    plt.axhline(sp382, c='r')
    plt.axhline(sp382_stats, c='m')
    plt.axhline(sp618, c='g')
    plt.axhline(sp618_stats, c='k')
    plt.fill_between(tsla_df.index, above618, below618, alpha=0.5, color='r')
    plt.fill_between(tsla_df.index, above382, below382, alpha=0.5, color='g')
    return namedtuple('golden', ['above618', 'below618', 'above382', 'below382'])(
        above618, below618, above382, below382)

golden = plot_golden()
plt.legend(['close', 'sp382', 'sp382_stats', 'sp618', 'sp618_stats'], loc='best')

#%%
print('理论上的最高盈利点: {}'.format(golden.above618 - golden.below382))

#%%
buy_rate = [0.20, 0.25, 0.30]
sell_rate = [0.70, 0.80, 0.90]

def find_percent_point(percent, y_org, want_max):
    """
    :param percent: 比例
    :param y_org: 价格序列
    :param want_max: 是否返回大的值
    """
    cs_max = y_org.max()
    cs_min = y_org.min()
    maxmin_imum = np.maximum if want_max else np.minimum
    return maxmin_imum(
        scs.scoreatpercentile(y_org, np.round(percent * 100, 1)),
        (cs_max - cs_min) * percent + cs_min)

result = []
result.append((0.382, 0.618, round(golden.above618 - golden.below382, 2)))

for (buy, sell) in product(buy_rate, sell_rate):
    profit_below = find_percent_point(buy, tsla_df.close, False)
    profit_above = find_percent_point(sell, tsla_df.close, True)
    result.append((buy, sell, round(profit_above - profit_below, 2)))

result = np.array(result)
result

#%%
# 1. 通过 scatter 点图
cmap = plt.get_cmap('jet', 20)
cmap.set_under('gray')
fig, ax = plt.subplots(figsize=(8, 5))
cax = ax.scatter(result[:, 0], result[:, 1], c=result[:, 2],
    cmap=cmap, vmin=np.min(result[:, 2]),
    vmax=np.max(result[:, 2]))
fig.colorbar(cax, label='max profit', extend='min')
plt.grid(True)
plt.xlabel('buy rate')
plt.ylabel('sell rate')
plt.show()

# 2. 通过 mpl_toolkits.mplot3d
fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
ax.view_init(30, 60)
ax.scatter3D(result[:, 0], result[:, 1], result[:, 2], c='r', s=50, cmap='spring')
ax.set_xlabel('buy rate')
ax.set_ylabel('sell rate')
ax.set_zlabel('max profit')
plt.show()
