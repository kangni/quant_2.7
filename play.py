#%%
import numpy as np
linspce_exp = np.linspace(0, 7, 10)
linspce_exp

#%%
stock_cnt = 200
view_days = 504
stock_day_change = np.random.standard_normal((stock_cnt, view_days))
print "original data"
print stock_day_change[:2, :5]
print "===================="
print "int data"
print stock_day_change[:2, :5].astype(int)
print "===================="
print "float data"
print np.around(stock_day_change[:2, :5], 2)
print "===================="
print "np.nan"
tmp_test = stock_day_change[:2, :5].copy()
tmp_test[0][0] = np.nan
print tmp_test
print "===================="
print "np.nan_to_num"
tmp_test = np.nan_to_num(tmp_test)
print tmp_test
print "===================="
print "bool"
mask = stock_day_change[:2, :5] > 0.5
print mask

#%%
print "original data"
print stock_day_change[:2, :5]
print stock_day_change[-2:, -5:]
print "===================="
print np.maximum(stock_day_change[:2, :5], stock_day_change[-2:, -5:])

#%%
print stock_day_change[:2, :5]
print "===================="
print np.diff(stock_day_change[:2, :5])
print np.diff(stock_day_change[:2, :5], axis=0)

#%%
print stock_day_change[-2:, -5:]
print "===================="
tmp_test = stock_day_change[-2:, -5:]
print tmp_test
print np.where(tmp_test > 0.5, 1, 0)
print np.where(np.logical_and(tmp_test > 0.5, tmp_test < 1), 1, 0)
print np.where(np.logical_or(tmp_test > 0.5, tmp_test < -0.5), 1, 0)

#%%
np.save('stock_day_change', stock_day_change)

#%%
stock_day_change = np.load('stock_day_change.npy')
stock_day_change.shape