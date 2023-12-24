import copy
import pylab
import random
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

import matplotlib.pyplot as plt
import pandas as pd
import dateutil
import random
import os

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

def get_stock(code,fq):
    # 获取并保存数据
    try:
        f = open('./storehouse/'+code+'.csv','r')
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
        stock_df.to_csv('./storehouse/'+code + '.csv')
    stock_df.index = pd.to_datetime(stock_df['date'])
    stock_df.index_col = 'date'
    return stock_df


# n-i，n+1
def Stock_range2(stock_df,td,count):
    list = stock_df['date'].tolist()
    try:
        index = list.index(td.strftime('%Y-%m-%d'))
        stock_range = stock_df[stock_df['date'].between(stock_df['date'][index-count+2],stock_df['date'][index+1])]
    except:
        stock_range = []
    # print(td,stock_range)
    return stock_range

# 后续考虑滑点
def get_today_data(Context,code):
    today = Context.dt.strftime('%Y-%m-%d')
    data = g.da[g.da['date']==today][:]
    return data

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

class G:
    pass
g = G()

# 回测函数
def run(Context):
    Init(Context)
    init_ben = benchmark(Context)
    init_cash = Context.cash
    plt_value = pd.DataFrame(index=pd.to_datetime(Context.date_range['trade_date']),columns=['value'])
    last_prize = {}

    for td in Context.date_range['trade_date']:
        # all steps num
        g.global_step += 1
        # print('Today:',td)
        Signal=2
        Context.dt = dateutil.parser.parse(str(td))

        if g.deal == 0:
            ma,ma20,Signal,action,next_state,reward = handle(Context,td,plt_value)
            if ma != 0:
                plt_value.loc[td,'ma'] = (ma-init_ben)/init_ben
                plt_value.loc[td,'ma20'] = (ma20-init_ben)/init_ben
        else:
            g.deal -= 1
        # print(g.deal)

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

        print('-'*30)
        print(td)
        print(f"\033[33m{'今日资产总额：'}\033[0m",round(Cash,2))
        print(f"\033[33m{'今日持仓：'}\033[0m",Context.positions)

        g.state = next_state
        next_action = g.agent.get_action(next_state)
        next_state = np.reshape(next_state, [1, 11])
        g.state = np.reshape(g.state, [1, 11])
        g.agent.train_model(g.state, action, round(100*reward,4), next_state, next_action, g.done)
        # every time step we do training
        g.score += reward
    

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
    # 结束标识
    g.done = True
    # plot
    plt_value['return'] = (plt_value['value']-init_cash) / init_cash
    plt_value['bench_self'] = (plt_value['value_ben'] - init_ben) / init_ben
    # plt_value[['return','benchmarker']].plot()
    plt_value[['return','bench_self','ma','ma20']].plot()
    try:
        plt.scatter(plt_value.index,plt_value['buy'],color='r',zorder=10)
        plt.scatter(plt_value.index,plt_value['sell'],color='g',zorder=10)
    except:
        pass
    #set_benchmark2(Context)
    plot_return(Context)
    plt_value.to_csv('./BSdata.csv')
    
    return plt_value['return'][-1]

def plot_return(Context):
    plt.legend()
    plt.axhline(y=0,c='grey',ls='--',lw=1,zorder=0)
    plt.grid(alpha=0.4)
    plt.xlabel(u'Date',fontsize=16)
    plt.ylabel(u'Return',fontsize=16)
    #plt.semilogy()
    #plt.show()
    plt.savefig('./test.png',bbox_inches='tight', dpi=256)
    plt.close()


def set_benchmark(Context,code):
    Context.benchmark = code

def set_benchmark2(Context):
    df = ak.stock_a_pe(symbol=Context.benchmark2)
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

def print_end():
    print('初始化完成')



# ---------------------------------------------------------------------------------------------------------------------------



# this is DeepSARSA Agent for the GridWorld
# Utilize Neural Network as q function approximator
class DeepSARSAgent:
    def __init__(self):
        self.load_model = False
        # actions which agent can do
        self.action_space = [-1, 0, 1]# 动作
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 11 # 环境状态等数据
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('./save_model/deep_sarsa_trained.h5')

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)
        else:
            # Predict the reward value based on the given state
            state = np.float32(state)
            q_values = self.model.predict(state)
            # print(q_values)
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done): 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print(state, action, reward, next_state, next_action)
        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        # like Q Learning, get maximum Q value at s'
        # But from target model
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])

        target = np.reshape(target, [1, self.action_size])
        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(state, target, epochs=1, verbose=0)

# 用户函数：--------------------------------------------------------------------------------

def Init(Context):
    g.code = '000099'
    g.da = get_stock(g.code,'hfq')
    g.init = set_benchmark(Context,g.code)
    g.c = 'close'
    g.deal = 0
    g.cash = Context.cash
    g.epi = 30
    g.scores, g.episodes = [], []
    g.global_step = 0
    g.done = False
    g.score = 0
    g.agent = DeepSARSAgent()
    g.state = []
    print_end()
    pass

# ---------------------------------------------
enable_hist_df = pd.read_csv('history.csv',parse_dates=['trade_date'])
enable_hist_df.columns = [
    'list',
    'trade_date',
]
# ---------------------------------------------

def handle(Context,td,value):
    list = Context.date_range['trade_date'].tolist()
    index = list.index(td)
    # 交易信号
    Signal=2
    # 是否要考虑停牌？
    history = Stock_range2(g.da,td,80)
    if len(history) == 0:
        print(f"\033[34m{'今日停牌，不交易'}\033[0m")
        return 0,0,Signal
    else:
        ma5 = history['close'][-11:-1].mean()
        ma20 = history['close'][-41:-1].mean()

        # get action for the current state and go one step in environment
        if len(g.state) == 0:
            g.state = history['close'].tolist()[-12:-2]
            g.state.append(0)
            sig = g.agent.get_action(g.state)
        else:
            sig = g.agent.get_action(g.state)
        reward = (history['close'][-1]-history['close'][-2])/history['close'][-2]
        # 输出的next state
        state = history['close'].tolist()[-10:]

        if len(Context.positions) == 0 :
            state.append(0)
        else:
            state.append(1)

        if Context.cash < 100 and sig==1:
            reward = 0
        if len(Context.positions) == 0 and sig==-1:
            reward = 0

        if sig==1:
            order_value(Context,g.code,Context.cash,g.c)
            Signal = 1
            print(Signal)
        elif sig==-1 and g.code in Context.positions:
            order_target(Context,g.code,0,g.c)
            Signal = 0
        
        # print(td,history['close'])
        
        
    return ma5,ma20,Signal,sig,state,reward


# Return = run(C)

#------------------
# 训练
if __name__ == "__main__":
    C = Context(100000,'2021-04-01','2021-06-01','hfq',"000099")
    g.agent = DeepSARSAgent()
    Init(C)
    for num in range(g.epi):

        C = Context(100000,'2021-04-01','2021-05-01','hfq',"000099")
        Return = run(C)

        g.scores.append(g.score)
        g.episodes.append(num)
        pylab.plot(g.episodes, g.scores, 'b')
        pylab.savefig("./save_graph/deep_sarsa_.png")
        print("episode:", num, "  score:", g.score, "global_step",
                g.global_step, "  epsilon:", g.agent.epsilon)

        if num % 100 == 0:
            g.agent.model.save_weights("./save_model/deep_sarsa.h5")

