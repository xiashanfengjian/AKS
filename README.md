# This is an akshare-based stock backtesting framework.
The stock data here is from akshare.
Aks provides the basic framework needed for backtesting, which you can modify and update to meet your personal needs.
My goal is to have it be highly customizable like vim while retaining its core functionality.
You can customize functions, change data sources, and even easily modify internal functions.

# Aks包括10个主要函数func：

1.trade_get():获取交易日数据

2.get_stock(code,fq):获取股票数据，code股票代码，fq：复权类型

3.Stock_range(stock_df,td,count):获取一定时间范围内的数据。依次输入，股票数据，当天日期，数量

4.get_today_data(Context,code):获取今天价格.依次输入，用户信息,股票代码

5.order_root(Context,today_price,code,amount,o_or_c):最底层的下单函数。依次输入，用户信息，今天价格，股票代码，交易数量，open_or_close，考虑了买入卖出佣金

6.order(Context,code,amount,o_or_c):下单函数之一。依次输入，用户信息，股票代码，交易数量，open_or_close

7.order_target(Context,code,amount,o_or_c):下单函数之一。依次输入，用户信息，股票代码，交易后的持仓数量，open_or_close

8.order_value(Context,code,value,o_or_c):下单函数之一。依次输入，用户信息，股票代码，交易后的持有现金，open_or_close

9.order_target_value(Context,code,value,o_or_c):下单函数之一。依次输入，用户信息，股票代码，交易后的持有现金，open_or_close

10.order_target_value(Context,code,value,o_or_c):

11.run(Context):用户信息，主要回测函数，按天回测

12.class Context:存了用户需要的信息

13.class G():非常重要，这是一个全局类，用于方便用户在初始化函数和策略函数里随心所欲地定义变量，这些变量都会被存在g的属性里

14.使用时需要导入aks.py,获取交易日信息，并新建Init函数记录取票代码等信息。新建handle函数，进行策略部署，然后调用run函数即可。在aks.py最后有一示例代码文件。

15. 2023.3.2更新了benchmark.和改善了图形界面。

16. 601318是测试用数据

17.增加了交易判断器g.deal, 当它为0时才运行策略

18.考虑了全佣的手续费，按照0.2%买入卖出

19.考虑了随机的交易滑点

20.2023-03-22修改了下单为0仍需交手续费的bug

21.2023-03-24可以设置沪深300，上证50等作为benchmark

22.plot_return()单独设置绘图参数

23.2023-10-15添加opt函数，用于资产组合优化，opt(f_list,l,td) 依次为股票代码列表，用于资产组合优化的时间序列长度，以及当天日期，返回买入标的列表，以及相应资产比例。

24.2023-10-16 Least_square(x,y) 最小二乘法回归直线，返回截距和斜率

25.2023-11-20 func.py存放了几种技术指标的计算函数

# Aks案例测试：
2023-11-20 RSRS择时
2023-12-24 强化学习demo

# AKS update aksm.py
2024-02-13 支持分钟级数据回测，包括：5min，30min，60min data文件夹存放了测试数据

