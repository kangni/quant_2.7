#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
linspce_exp = np.linspace(0, 7, 10)
linspce_exp

#%%
stock_cnt = 200
view_days = 504
stock_day_change = np.random.standard_normal((stock_cnt, view_days))
stock_example_a = stock_day_change[:2, :5]
print 'original data'
print stock_example_a
print '===================='
print 'int data'
print stock_example_a.astype(int)
print '===================='
print 'float data'
print np.around(stock_example_a, 2)
print '===================='
print 'np.nan'
tmp_test = stock_example_a.copy()
tmp_test[0][0] = np.nan
print tmp_test
print '===================='
print 'np.nan_to_num'
tmp_test = np.nan_to_num(tmp_test)
print tmp_test
print '===================='
print 'bool'
mask = stock_example_a > 0.5
print mask

#%%
stock_example_b = stock_day_change[-2:, -5:]
print 'original data'
print stock_example_a
print stock_example_b
print '===================='
print np.maximum(stock_example_a, stock_example_b)

#%%
print stock_example_a
print '===================='
print np.diff(stock_example_a)
print np.diff(stock_example_a, axis=0)

#%%
tmp_test = stock_example_b
print tmp_test
print np.where(tmp_test > 0.5, 1, 0)
print np.where(np.logical_and(tmp_test > 0.5, tmp_test < 1), 1, 0)
print np.where(np.logical_or(tmp_test > 0.5, tmp_test < -0.5), 1, 0)

#%%
np.save('stock_day_change', stock_day_change)

#%%
stock_day_change = np.load('stock_day_change.npy')
stock_day_change.shape

#%%
stock_day_change_four = stock_day_change[:4, :4]
print stock_day_change_four
print '===================='
print '4只股票在4天内的表现'
print '最大涨幅 {}'.format(np.max(stock_day_change_four, axis=1))
print '最大跌幅 {}'.format(np.min(stock_day_change_four, axis=1))
print '振幅幅度 {}'.format(np.std(stock_day_change_four, axis=1))
print '平均涨跌 {}'.format(np.mean(stock_day_change_four, axis=1))
print '===================='
print '某一交易日4只股票的表现'
print '最大涨幅 {}'.format(np.max(stock_day_change_four, axis=0))
print '最大涨幅股票 {}'.format(np.argmax(stock_day_change_four, axis=0))

#%%
a_investor = np.random.normal(loc=100, scale=50, size=(10000, 1))
b_investor = np.random.normal(loc=100, scale=20, size=(10000, 1))

a_mean = a_investor.mean()
a_std = a_investor.std()
a_var = a_investor.var()

b_mean = b_investor.mean()
b_std = b_investor.std()
b_var = b_investor.var()

print 'a 交易者期望 {0:.2f} 元，标准差 {1:.2f}，方差 {2:.2f}'.format(a_mean, a_std, a_var)
print 'b 交易者期望 {0:.2f} 元，标准差 {1:.2f}，方差 {2:.2f}'.format(b_mean, b_std, b_var)

#%%
plt.plot(a_investor)
plt.axhline(a_mean + a_std, color='r')
plt.axhline(a_mean, color='y')
plt.axhline(a_mean -  a_std, color='g')
print 'a investor'

#%%
plt.plot(b_investor)
plt.axhline(b_mean + b_std, color='r')
plt.axhline(b_mean, color='y')
plt.axhline(b_mean - b_std, color='g')
print 'b investor'

#%%
first_stock = stock_day_change[0]
stock_mean = first_stock.mean()
stock_std = first_stock.std()
print '股票 0 mean 均值期望:{:.3f}'.format(stock_mean)
print '股票 0 std 振幅标准差:{:.3f}'.format(stock_std)
plt.hist(first_stock, bins=50, normed=True)
fit_linspece = np.linspace(first_stock.min(), first_stock.max())
pdf = scs.norm(stock_mean, stock_std).pdf(fit_linspece)
plt.plot(fit_linspece, pdf, lw=2, c='r')