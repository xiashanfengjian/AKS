# basic func
def GaoKai(history,num):
    b = history['open'][-num]-history['close'][-(num+1)]
    if b>0:
        b=1
    else:
        b=-1
    return b

def Gaoend(history,num):
    b = history['open'][-num]-history['close'][-(num+1)]
    if b<0:
        b=1
    else:
        b=-1
    return b






def TwoCrows(history,key):
    condition1 = history['close'][-3]-history['open'][-3] # >0 长阳
    condition2_1 = history['open'][-2]-history['close'][-3] # >0 高开
    condition2_2 = history['close'][-2]-history['open'][-2] # <0 收阴
    condition3_1 = history['open'][-1]-history['close'][-2] # >0 高开
    condition3_2 = history['close'][-1]-history['open'][-1] # <0 收阴
    condition3_3 = history['close'][-1]-history['close'][-2] # <0 低于前日收盘价
    reslut = 0
    if condition1>0:
        if condition2_1>0 and condition2_2<0:
            if condition3_1>0 and condition3_2<0 and condition3_3<0:
                reslut == -1
    return reslut

def ThreeCrows(history,key):
    condition1_1 = history['close'][-3]-history['open'][-3] # <0 阴
    condition1_2 = history['close'][-3]/history['low'][-3] - 1.001 # <0 收盘接近最低
    condition1_3 = (history['open'][-3]-history['open'][-4])*(history['open'][-3]-history['close'][-4])#<0 open在前一日实体内
    condition2_1 = history['close'][-2]-history['open'][-2] # <0 阴
    condition2_2 = abs((history['close'][-2]-history['low'][-2])/(history['close'][-2]-history['open'][-2])) - 0.1 # <0 收盘接近最低
    condition2_3 = (history['open'][-2]-history['open'][-3])*(history['open'][-3]-history['close'][-4])#<0 open在前一日实体内
    condition3_1 = history['close'][-1]-history['open'][-1] # <0 阴
    condition3_2 = abs((history['close'][-1]-history['low'][-1])/(history['close'][-1]-history['open'][-1])) - 0.1 # <0 收盘接近最低
    condition3_3 = (history['open'][-1]-history['open'][-2])*(history['open'][-3]-history['close'][-4])#<0 open在前一日实体内
    reslut = 0
    if condition1_1<0 and condition1_2<0 and condition1_3<0:
        if condition2_1<0 and condition2_2<0 and condition2_3<0:
            if condition3_1<0 and condition3_2<0 and condition3_3<0:
                reslut == 1
    return reslut

def ThreeInsideUpDown(history,key):
    condition1 = history['close'][-3]-history['open'][-3] # <0 收阴
    condition2_1 = history['close'][-2]-history['open'][-2] # >0 阳
    condition2_2 = (history['open'][-2]-history['open'][-3]) * (history['open'][-3]-history['close'][-4])#<0 open在前一日实体内
    condition2_3 = (history['close'][-2]-history['open'][-3])*(history['close'][-3]-history['close'][-4])#<0 open在前一日实体内
    condition3_1 = history['close'][-1]-history['open'][-1] # >0 阳
    condition3_2 = history['close'][-1]-history['open'][-3] # >0 第3天close大于第1天open
    reslut = 0
    if condition1<0:
        if condition2_1>0 and condition2_2<0 and condition2_3<0:
            if condition3_1>0 and condition3_2>0:
                reslut == 1
    return reslut

def ThreeLineStrike(history,key):
    condition1_1 = history['close'][-4]-history['open'][-4] # >0 阳
    condition1_2 = history['close'][-4]-history['close'][-5] # >0 高于前日收盘价
    condition2_1 = history['close'][-3]-history['open'][-3] # >0 阳
    condition2_2 = history['close'][-3]-history['close'][-4] # >0 高于前日收盘价
    condition3_1 = history['close'][-2]-history['open'][-2] # >0 阳
    condition3_2 = history['close'][-2]-history['close'][-3] # >0 高于前日收盘价
    condition4_1 = history['open'][-1]-history['close'][-2] # >0 高开
    condition4_2 = history['close'][-1]-history['open'][-2] # <0 第4天close小于第1天open
    reslut = 0
    if condition1_1>0 and condition1_2>0:
        if condition2_1>0 and condition2_2>0:
            if condition3_1>0 and condition3_2>0:
                if condition4_1>0 and condition4_2<0:
                    reslut == -1
    return reslut
    
def ThreeStarsInSouth(history,key):
    reslut = 0
    return reslut
