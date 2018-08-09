import matplotlib.pyplot as plt

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