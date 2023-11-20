import akshare as ak
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dateutil
import datetime
import random
import os

# 计算杨辉三角
def triangle(n):
    N = [1]
    for i in range(n):  #打印n行
        N.append(0)
        N = [N[k] + N[k-1] for k in range(i+2)]
    N = np.array(N)
    return N

def EXp_N(L,n):
    l = np.linspace(0,L,n)
    N = np.exp(-l)
    return N

if True:
    matplotlib.rcParams['axes.linewidth']       = 1
    matplotlib.rcParams['xtick.major.size']     = 6
    matplotlib.rcParams['xtick.major.width']    = 1
    matplotlib.rcParams['xtick.minor.size']     = 6
    matplotlib.rcParams['xtick.minor.width']    = 1
    matplotlib.rcParams['ytick.major.size']     = 6
    matplotlib.rcParams['ytick.major.width']    = 1
    matplotlib.rcParams['ytick.minor.size']     = 6
    matplotlib.rcParams['ytick.minor.width']    = 1
    matplotlib.rcParams['ytick.direction']     = 'in'
    matplotlib.rcParams['xtick.direction']    = 'in'

    matplotlib.rcParams['xtick.major.pad']      = 10
    matplotlib.rcParams['ytick.major.pad']      = 10

    matplotlib.rcParams['mathtext.default']  = 'regular'
#-- End

# API for aks
# 个股信息查询
def search_code(code,print_or_not):
    try:
        stock = ak.stock_individual_info_em(symbol=code)
    except:
        print('Not found')
    if print_or_not == 1:
        print(stock)
    return stock
# 所有A股上市公司的实时行情数据
def all_in_A():
    all = ak.stock_zh_a_spot_em()
    return all

def trade_get():
    enable_hist_df = ak.tool_trade_date_hist_sina()
    enable_hist_df.columns = ['list','trade_date',]
    enable_hist_df['trade_date'] = pd.to_datetime(enable_hist_df['trade_date'])
    return enable_hist_df

def getpool(code):
    pool = ak.index_stock_cons(symbol=code)
    return pool

def get_stock(code,fq):
    # 获取并保存数据
    try:
        f = open('../storehouse/'+code+'.csv','r')
        stock_df = pd.read_csv(f)
    except:
        print(1)
        stock_df = ak.stock_zh_a_hist(symbol=code, adjust=fq).iloc[:, :6]
        stock_df.columns = [
            'date',
            'open',
            'close',
            'high',
            'low',
            'volume',
        ]
        stock_df.to_csv('../storehouse/'+code + '.csv')
    stock_df.index = pd.to_datetime(stock_df['date'])
    stock_df.index_col = 'date'
    return stock_df

# 获取一定范围内的数据
# n-i，n
def Stock_range(stock_df,td,count):
    list = stock_df['date'].tolist()
    try:
        index = list.index(td.strftime('%Y-%m-%d'))
        stock_range = stock_df[stock_df['date'].between(stock_df['date'][index-count+1],stock_df['date'][index-1+1])]
    except:
        stock_range = []
    # print(td,stock_range)
    return stock_range
# n-i，n-1
def Stock_range2(stock_df,td,count):
    list = stock_df['date'].tolist()
    try:
        index = list.index(td.strftime('%Y-%m-%d'))
        stock_range = stock_df[stock_df['date'].between(stock_df['date'][index-count],stock_df['date'][index-1])]
    except:
        stock_range = []
    # print(td,stock_range)
    return stock_range

# 预留，上面函数似乎过于复杂，留待简化
def Stock_hist(td,code,count):
    return

# 后续考虑滑点
def get_today_data(Context,code):
    today = Context.dt.strftime('%Y-%m-%d')
    da = get_stock(code,Context.fq)
    data = da[da['date']==today][:]
    return data

# 已经考虑停牌，后续考虑分红
def order_root(Context,today_price,code,amount,o_or_c):
    
    if len(today_price) == 0:
        print(f"\033[33m{'今日停牌！'}\033[0m")
        return
    ymd = today_price['date'][0]
    # 应在底层下单函数中考虑滑点
    logi = random.choice([-1,1])
    today_price = today_price[o_or_c][0]*(1+logi*0.5/100)
    if amount>0:
        if Context.cash - amount*today_price < 0:
            amount = int(int(Context.cash/today_price)/100)
            if amount == 0:
                print(f"\033[31m{ymd}\033[0m",f"\033[33m{':现金严重不足，无法买入！！！'}\033[0m")
            else:
                amount = amount*100
                print(f"\033[31m{ymd}\033[0m",f"\033[33m{':现金不足，已帮您调整为%d' % (amount)}\033[0m")
        else:
            amount = int(amount/100)
            amount = amount*100
            print(f"\033[31m{ymd}\033[0m",f"\033[33m{':现金充足，已做整数调整，调整后买入%d' % (amount)}\033[0m")
        
    else:
        if amount+Context.positions.get(code,0)<=0:
            if amount+Context.positions.get(code,0)<0:
                amount = -Context.positions.get(code,0)
                print(f"\033[31m{ymd}\033[0m",f"\033[33m{':持仓不足，全仓卖出！'}\033[0m")
            elif Context.positions.get(code,0)==0:
                amount = -Context.positions.get(code,0)
                print(f"\033[31m{ymd}\033[0m",f"\033[33m{':持仓为0，无法卖出！'}\033[0m")
            elif amount+Context.positions.get(code,0)==0:
                amount = -Context.positions.get(code,0)
                print(f"\033[31m{ymd}\033[0m",f"\033[33m{':持仓刚好，全仓卖出！'}\033[0m")        
        
        else:
            amount = int(amount/100)
            amount = amount*100
            print(f"\033[31m{ymd}\033[0m",f"\033[33m{':持仓充足，已做整数调整，调整后卖出%d' % -amount}\033[0m")

            
    if amount != 0:
        if amount>0:
            # 买入手续费，按全佣上限0.2%
            service = abs(amount)*today_price*0.2/100
        else:
            # 卖出手续费，按全佣上限0.2%
            service = abs(amount)*today_price*0.2/100
        if service <= 5:
            service = 5
        Context.cash -= service
    else:
        service = 0
    
    Context.cash -= amount*today_price
    
    print('          ',f"\033[30m{'Service charge:'}\033[0m",round(service,2))
    Context.positions[code] = Context.positions.get(code,0)+amount
    # print(Context.positions)
    if Context.positions[code] == 0:
        del Context.positions[code]
    return

def order(Context,code,amount,o_or_c,today_price = 0):
    if today_price == 0:
        today_price = get_today_data(Context,code)
    order_root(Context,today_price,code,amount,o_or_c)

def order_target(Context,code,amount,o_or_c,today_price = 0):
    if amount<0:
        print('数量不能为负，已调整为0')
    if today_price == 0:
        today_price = get_today_data(Context,code)
    hold_amount = Context.positions.get(code,0)
    delta_amount = amount - hold_amount
    order_root(Context,today_price,code,delta_amount,o_or_c)

def order_value(Context,code,value,o_or_c,today_price = 0):
    if today_price == 0:
        today_price = get_today_data(Context,code)
    amount = int(value/today_price[o_or_c][0])
    order_root(Context,today_price,code,amount,o_or_c)

def order_target_value(Context,code,value,o_or_c,today_price = 0):
    if value<0:
        print('价值不能为负，已调整为0')

    if today_price == 0:
        today_price = get_today_data(Context,code)
    hold_value = Context.positions.get(code,0)*today_price[o_or_c][0]
    delta_value = value - hold_value
    order_value(Context,code,delta_value,o_or_c)

# 非常重要，这是一个全局类，用于方便用户在初始化函数和策略函数里随心所欲地定义变量，这些变量都会被存在g的属性里
class G:
    pass
g = G()
# 交易判断器
g.deal = 0
g.cc = 0

# 回测函数
def run(Context):
    Init(Context)
    init_ben = benchmark(Context)
    init_cash = Context.cash
    plt_value = pd.DataFrame(index=pd.to_datetime(Context.date_range['trade_date']),columns=['value'])
    last_prize = {}

    for td in Context.date_range['trade_date']:
        g.cc += 1
        # print('Today:',td)
        Signal=2
        Context.dt = dateutil.parser.parse(str(td))

        
        Cash = Context.cash
        
        for stock_code in Context.positions:
            today_p = get_today_data(Context,stock_code)
            if len(today_p) == 0:
                p = last_prize[stock_code]
            else:
                p = today_p['close'][0]
                last_prize[stock_code] = p
            Cash += p*Context.positions[stock_code]
            
        plt_value.loc[td,'value'] = Cash

        # handle(Context,td)
        # 判断今日是否交易
        # print(td,g.deal)
        if g.deal == 0:
            ma,ma20,Signal = handle(Context,td,plt_value)
            if ma != 0:
                plt_value.loc[td,'ma'] = (ma-init_ben)/init_ben
                plt_value.loc[td,'bate'] = ma20
        else:
            g.deal -= 1
        # print(g.deal)
        if len(Context.positions) != 0:
            g.days += 1

        # benchmark
        today_p_ben = get_today_data(Context,Context.benchmark)
        if len(today_p_ben) == 0:
            p2 = last_prize[Context.benchmark]
        else:
            p2 = today_p_ben['close'][0]
            last_prize[Context.benchmark] = p2
        plt_value.loc[td,'value_ben'] = p2
        # print(Cash)

        # 记录交易信号
        if Signal==1:
            plt_value.loc[td,'buy'] = (p2-init_ben)/init_ben
        elif Signal==0:
            plt_value.loc[td,'sell'] = (p2-init_ben)/init_ben

    # plot
    plt_value['return'] = (plt_value['value']-init_cash) / init_cash
    plt_value['bench_self'] = (plt_value['value_ben'] - init_ben) / init_ben
    # plt_value[['return','benchmarker']].plot()
    plt_value[['return','bench_self','ma','bate']].plot()
    plt.scatter(plt_value.index,plt_value['buy'],color='r',zorder=10)
    plt.scatter(plt_value.index,plt_value['sell'],color='g',zorder=10)
    remean = 100*(plt_value['return'][-10:].mean()/len(plt_value['return'][:]-10))
    sharp = (plt_value['return'][:-2].std())
    plt.figtext(0.2, 0.4, str(round(remean,2)), fontsize = 12)
    plt.figtext(0.2, 0.3, str(round(sharp,2)), fontsize = 12)
    print('Return:',sharp,plt_value['return'][-1],round(remean,2))#,'Sharp:',round(sharp,2))

    #set_benchmark2(Context)
    plot_return(Context)
    if np.sum(plt_value['return'][:])< 0.03:
    #if plt_value['return'][-1]< 0.03:
        return -1
    elif np.sum(plt_value['return'][:])> 0.1:
    #elif plt_value['return'][-1]> 0.2:
        return 1
    else:
        return 0
    

    
def plot_return(Context):
    plt.legend()
    plt.axhline(y=0,c='grey',ls='--',lw=1,zorder=0)
    plt.grid(alpha=0.4)
    plt.xlabel(u'Date',fontsize=16)
    plt.ylabel(u'Return',fontsize=16)
    #plt.semilogy()
    plt.show()
    plt.savefig('../check/'+Context.date_start+'.png',bbox_inches='tight', dpi=128)
    plt.close()


def set_benchmark(Context,code):
    Context.benchmark = code

def set_benchmark2(Context):
    df = ak.index_zh_a_hist(symbol=Context.benchmark2,period="daily")
    df.index = pd.to_datetime(df['date'])
    try:
        df = df['close'][Context.date_start:Context.date_end]
        df = (df[:]-df[0])/df[:]
        plt.plot(df.index,df,label=Context.benchmark2)
    except:
        print('无法获取benchmark')
        return

def benchmark(Context):
    stock_df = get_stock(Context.benchmark,Context.fq)
    stock_range = stock_df[stock_df['date'].between(Context.date_start,Context.date_end)]
    if len(stock_range['close'])==0:
        print(f"\033[31m{'无法获取历史数据，回测失败！起始时间为'}\033[0m",stock_df['date'][0])
        os.kill()
    # print(stock_range)
    else:
        init_r = stock_range['close'][0]
        return init_r

def Tuoshen(index,value,Context):
    if index > 10 and g.deal < 5 :
            value = list(value['value'])
            # print(value[index-10])
            tt = (value[index-1]-value[index-5])/value[index-5]
            # print(tt,index)
            if  tt > 0.2 or tt < -0.2:
                print('警戒线',tt)
                
                if len(Context.positions) != 0 and g.deal ==0:
                    print(list(Context.positions.keys()))
                    for code in list(Context.positions.keys()):
                        order_target(Context,code,0,g.c)
                if  tt > 0.1:
                    g.deal = 1
                else:
                    g.deal = 10

class Context:
    def __init__(self, cash, date_start, date_end, fq, bench):
        self.cash = cash
        self.date_start = date_start
        self.date_end = date_end
        self.fq = fq
        self.positions = {}
        self.benchmark = '00'
        self.benchmark2 = bench
        self.date_range = enable_hist_df[enable_hist_df['trade_date'].between(date_start,date_end)]
        self.dt = None  # dateutil.parser.parse(date_start)

def print_end():
    print('初始化完成')


def Least_square(x,y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
    A = np.mat([[N,sumx],[sumx,sumx2]])
    b = np.array([sumy,sumxy])
      
    return np.linalg.solve(A,b)

def handle(Context,td,value):
    list_ = Context.date_range['trade_date'].tolist()
    index = list_.index(td)

    # 交易信号
    Signal=2
    weekday = td.weekday()

    # 考虑停牌
    history = Stock_range2(get_stock(g.code,Context.fq),td,110)
    if len(history) == 0:
        print(f"\033[34m{'今日停牌，不交易'}\033[0m")
        return 0,0,Signal
    else:
        ma20 = history['close'][-50:].mean()
        ma5 = history['close'][-10:].mean()
    
        yh = history['high'][-18:]
        yl = history['low'][-18:]
        c,bh = Least_square(yl,yh)
        # print(bh)
        if bh>1 and g.code not in Context.positions and g.deal==0:
            order_value(Context,g.code,Context.cash,g.c)
            g.p = get_today_data(Context,g.code)
            Signal = 1
            
        elif bh<0.9 and g.code in Context.positions and g.deal==0:
            order_target(Context,g.code,0,g.c)
            g.days = 0
            Signal = 0
    
    return ma5,bh,Signal




#(history['close'][-1]-history['close'][-2])/history['close'][-2]<=-0.05 and 

# 用户函数：--------------------------------------------------------------------------------

def Init(Context):
    # g.code = '601390'
    g.init = set_benchmark(Context,g.code)
    g.c = 'close'
    g.p = 0
    g.days = 0
    
    g.cash = Context.cash
    print_end()
    pass

# ---------------------------------------------
enable_hist_df = pd.read_csv('history.csv',parse_dates=['trade_date'])
enable_hist_df.columns = [
    'list',
    'trade_date',
]
# ---------------------------------------------

# Test
# f = open('../storehouse/601390.csv','r')
g.code = '601390'
# g.code = "601816"
get_stock(g.code,'hfq')
C = Context(500000,'2016-10-01','2023-09-15','hfq',"000300")
a = run(C)

