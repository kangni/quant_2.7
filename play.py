#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scs

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

#%%
stock_example_b = stock_day_change[-2:, -5:]
print('original data')
print(stock_example_a)
print(stock_example_b)
print('====================')
print(np.maximum(stock_example_a, stock_example_b))

#%%
print(stock_example_a)
print('====================')
print(np.diff(stock_example_a))
print(np.diff(stock_example_a, axis=0))

#%%
tmp_test = stock_example_b
print(tmp_test)
print(np.where(tmp_test > 0.5, 1, 0))
print(np.where(np.logical_and(tmp_test > 0.5, tmp_test < 1), 1, 0))
print(np.where(np.logical_or(tmp_test > 0.5, tmp_test < -0.5), 1, 0))

#%%
np.save('stock_day_change', stock_day_change)

#%%
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

#%%
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

#%%
cheat_moneys = [casino(0.4, commission=0) for _ in np.arange(0, gamblers)]
plt.hist(cheat_moneys, bins=30)

#%%
commission_moneys = [casino(0.5, commission=0.01) for _ in np.arange(0, gamblers)]
plt.hist(commission_moneys, bins=30)

#%%
moneys = [casino(0.5, commission=0.01, win_once=1.02, loss_once=0.98) for _ in np.arange(0, gamblers)]
plt.hist(moneys, bins=30)

#%%
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
import abupy
from abupy import ABuMarketDrawing 
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
from abupy import ABuSymbolPd
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