This is an akshare-based stock backtesting framework.
The stock data here is from akshare.
Aks provides the basic framework needed for backtesting, which you can modify and update to meet your personal needs.
If you want to change to another data source, you can easily modify it.

# 包括10个主要函数func：

1.trade_get():获取交易日数据

2.get_stock(code,fq):获取股票数据，code股票代码，fq：复权类型

3.Stock_range(stock_df,Context,enable_hist_df,td,count):获取一定时间范围内的数据。依次输入，股票数据，用户信息，交易日数据，当天日期，数量

4.get_today_data(Context,code):获取今天价格.依次输入，用户信息,股票代码

5.order_root(Context,today_price,code,amount,o_or_c):最底层的下单函数。依次输入，用户信息，今天价格，股票代码，交易数量，open_or_close

6.order(Context,code,amount,o_or_c):下单函数之一。依次输入，用户信息，股票代码，交易数量，open_or_close

7.order_target(Context,code,amount,o_or_c):下单函数之一。依次输入，用户信息，股票代码，交易后的持仓数量，open_or_close

8.order_value(Context,code,value,o_or_c):下单函数之一。依次输入，用户信息，股票代码，交易后的持有现金，open_or_close

9.order_target_value(Context,code,value,o_or_c):下单函数之一。依次输入，用户信息，股票代码，交易后的持有现金，open_or_close

10.order_target_value(Context,code,value,o_or_c):

11.run(Context):用户信息，主要回测函数，按天回测

12.class Context:存了用户需要的信息

13.class G():非常重要，这是一个全局类，用于方便用户在初始化函数和策略函数里随心所欲地定义变量，这些变量都会被存在g的属性里

14.使用时需要导入aks.py,获取交易日信息，并新建Init函数记录取票代码等信息。新建handle函数，进行策略部署，然后调用run函数即可。在aks.py最后有一示例代码文件。

# # ---------------------------------------------
# enable_hist_df = pd.read_csv('history.csv',parse_dates=['trade_date'])
# enable_hist_df.columns = [
#     'list',
#     'trade_date',
# ] # 这里的交易日数据需要用户使用trade_get()自行获取并保存
# # ---------------------------------------------


# 用户函数：示例如下

# def Init(Context):
#     g.code = '601318'
#     g.a = 5
#     g.b = 1000
#     g.c = 'open'
#     pass

# def handle(Context,td):
#     # 是否要考虑停牌？
#     history = Stock_range(get_stock(g.code,Context.fq),Context,enable_hist_df,td,60)
#     if len(history) == 0:
#         print(f"\033[34m{'今日停牌，不交易'}\033[0m")
#     else:
#         ma5 = history['close'][-5:].mean()
#         ma20 = history['close'][:].mean()
#         if ma5>ma20 and g.code not in Context.positions:
#             order_value(Context,g.code,Context.cash,g.c)
#         elif ma5<ma20 and g.code in Context.positions:
#             order_target(Context,g.code,0,g.c)
#     return


# def benchmark():

#     pass

# # Test
# C = Context(100000,'2010-01-01','2017-01-01','hfq')
# run(C)
