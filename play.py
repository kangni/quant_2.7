#%%
import numpy as np
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