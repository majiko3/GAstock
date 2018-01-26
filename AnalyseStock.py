
# coding: utf-8

# In[2]:


#coding: utf-8
import numpy as np
import os
import pandas as pd
import math
import subprocess
#subprocess.run(['jupyter', 'nbconvert',
#                '--to', 'python', 'AnalyseStock.ipynb'])
from Constants import *


class Stock:

    def read_df(self, fr_name):
        """
        Read stock data.
        楽天証券MARKETSPEEDから得られる時系列日足データを想定
        fr_name: Reading file name
        """
        #object形式で一括読み込み
        df = pd.read_csv(fr_name, dtype='object')
        #floatに変換
        date = df.apply(lambda x: pd.to_datetime(x['日付'].replace('/','-')), axis=1)
        open_ = df.apply(lambda x: float(x['始値'].replace(',','')), axis=1)
        high_ = df.apply(lambda x: float(x['高値'].replace(',','')), axis=1)
        low_ = df.apply(lambda x: float(x['安値'].replace(',','')), axis=1)
        close_ = df.apply(lambda x: float(x['終値'].replace(',','')), axis=1)
        flu_ = df.apply(lambda x: float(x['前日比'].replace(',','')), axis=1)
        dma25_ = df.apply(lambda x: float(x['25DMA'].replace('-','0').replace(',','')), axis=1)

        self.stock_df = pd.DataFrame({'Date': date,
                                      'Open': open_,
                                      'High': high_,
                                      'Low': low_,
                                      'Close': close_,
                                      'Fluctuation': flu_,
                                      '25DMA': dma25_
                                     },
                                     columns = ['Date', 'Open', 'High', 'Low',
                                                'Close', 'Fluctuation', '25DMA']
                                    )


    def set_newdf(self, fw_name):
        """
        Add indicators to stock data and write new file.
        fw_name: Writing file name
        """
        if os.path.isfile(fw_name) == True:
            print('Use the file which has already existed.')
            df = pd.read_csv(fw_name)
            self.new_stock_df = pd.DataFrame({'Date': df['Date'],
                                              'Open': df['Open'],
                                              'High': df['High'],
                                              'Low': df['Low'],
                                              'Close': df['Close'],
                                              'Fluctuation': df['Fluctuation'],
                                              '25DMA': df['25DMA'],
                                              'RSI': df['RSI'],
                                              'Psycho': df['Psycho'],
                                              'Stochas': df['Stochas'],
                                              '25Deviation': df['25Deviation'],
                                              '25DevNormal': df['25DevNormal']
                                             },
                                             columns = ['Date', 'Open', 'High', 'Low',
                                                        'Close', 'Fluctuation', '25DMA',
                                                        'RSI', 'Psycho', 'Stochas',
                                                        '25Deviation', '25DevNormal']
                                            )

            return self.new_stock_df

        self.new_stock_df = pd.DataFrame({'Date': self.stock_df['Date'],
                                          'Open': self.stock_df['Open'],
                                          'High': self.stock_df['High'],
                                          'Low': self.stock_df['Low'],
                                          'Close': self.stock_df['Close'],
                                          'Fluctuation': self.stock_df['Fluctuation'],
                                          '25DMA': self.stock_df['25DMA'],
                                          'RSI': self.RSI(),
                                          'Psycho': self.Psycho(),
                                          'Stochas': self.Stochastics(),
                                          '25Deviation': self.DMAdeviation(),
                                          '25DevNormal': self.Normalization(self.DMAdeviation(),
                                                                             self.DMAdeviation().max(),
                                                                             self.DMAdeviation().min())
                                         },
                                         columns = ['Date', 'Open', 'High', 'Low',
                                                    'Close', 'Fluctuation', '25DMA',
                                                    'RSI', 'Psycho', 'Stochas',
                                                    '25Deviation', '25DevNormal']
                                        )

        self.new_stock_df.to_csv(fw_name)

        return self.new_stock_df


    def RSI(self):
        """
        Calculation RSI.
        return: pd.Series(rsi)
        """
        rsi = []
        d = 14 #Parameter, Short:9, Long:14
        flu_ = self.stock_df['Fluctuation']
        #for i in range(len(flu_)):
            #try:
            #    flu_data_plus = []
            #    flu_data_minus = []
            #    flu_data = []
            #    flu_data = flu_[i:i+d] #d日分の前日比の変動幅を取得
            #    #df["Fluctuation"]はSeries型。リストでない。range(i,i+d)にする。
            #    for j in range(i,i+d):
            #        if flu_data[j] >= 0:
            #            flu_data_plus.append(flu_data[j])
            #        else:
            #            flu_data_minus.append(abs(flu_data[j]))
            #    ave_flu_data_plus = sum(flu_data_plus) / d
            #    ave_flu_data_minus = sum(flu_data_minus) / d
            #    rsi_value = ave_flu_data_plus / (ave_flu_data_plus+abs(ave_flu_data_minus))
            #    rsi.append(rsi_value)
            #except KeyError:
            #    rsi.append(np.nan)
        for i in range(len(flu_)-d+1):
            #リスト内包表記でも良いがKeyErrorが発生しないので，
            #最初のforloopのrangeをlen(flu_range)-dといったように変える必要がある
            #np.nanは代入しなくてもいいよね多分
            flu_data = flu_[i:i+d]
            flu_data_plus  = [x for x in flu_data if x>=0]
            flu_data_minus = [x for x in flu_data if x<0]
            ave_flu_data_plus = sum(flu_data_plus) / d
            ave_flu_data_minus = sum(flu_data_minus) / d
            rsi_value = ave_flu_data_plus / (ave_flu_data_plus+abs(ave_flu_data_minus))
            rsi.append(round(rsi_value, 4))

        return pd.Series(rsi)


    def Psycho(self):
        """
        Calculation Pychological-line.
        return: pd.Sereies(psycho)
        """
        psycho = []
        d = 12 #Parameter
        flu_ = self.stock_df['Fluctuation']
        for i in range(len(flu_)-d+1):
            flu_data = flu_[i:i+d] #d日分の前日比の変動幅を取得
            flu_data_plus = [x for x in flu_data if x>=0]
            psycho_value = len(flu_data_plus)/d
            psycho.append(round(psycho_value, 4))

        return pd.Series(psycho)


    def Stochastics(self):
        """
        Calculation Stochastics.
        return: pd.Sereies(stochas)
        """
        stochas = []
        d = 9 #Parameter 5, 9, 14
        close_ = self.stock_df['Close']
        high_ = self.stock_df['High']
        low_ = self.stock_df['Low']
        for i in range(len(close_)-d+1):
            high_data = high_[i:i+d]
            low_data = low_[i:i+d]
            max_ = max(high_data)
            min_ = min(low_data)
            stochas_value = (close_[i]-min_)/(max_-min_)
            stochas.append(round(stochas_value, 4))

        return pd.Series(stochas)


    def DMAdeviation(self):
        """
        Calculation difference from 25 Days Moving Average.
        return: pd.Sereies(dma25)
        """
        dev = []
        d = 25 #Parameter
        dma25_ = self.stock_df['25DMA']
        close_ = self.stock_df['Close']
        for i in range(len(dma25_)-d+1):
            dev_value = (close_[i]-dma25_[i])/dma25_[i]
            dev.append(round(dev_value, 4))

        return pd.Series(dev)


    def Normalization(self, x, X_max, X_min):
        """
        Normalization function.
        """
        y = (x-X_min)/(X_max-X_min)
        return round(y, 4)


    def maxDev(self):
        return self.new_stock_df['25Deviation'].max()


    def minDev(self):
        return self.new_stock_df['25Deviation'].min()


class Trading:

    def __init__(self, df):
        self.open_ = df['Open']
        self.close_ = df['Close']
        self.rsi_ = df['RSI']
        self.psycho_ = df['Psycho']
        self.stochas_ = df['Stochas']
        self.devnor_ = df['25DevNormal']


    def onlyLong(self, ind):
        """
        買って売るだけ，空売りはしない
        return: Paformance and position list.
        """
        date_len = len(self.open_) #データの長さ
        L_posi = 0 #Positionの有無
        pafo_list = [] #損益率
        posi = []
        for i in reversed(range(1,date_len)):
            if L_posi == 0:

                #買い売り共に点灯することがあれば評価しない
                if ind[0] >= ind[1]:
                    pafo_list = []
                    break
                elif ind[2] >= ind[3]:
                    pafo_list = []
                    break
                elif ind[4] >= ind[5]:
                    pafo_list = []
                    break
                elif ind[6] >= ind[7]:
                    pafo_list = []
                    break
                else:
                    pass

                #各指標によるシグナルを計算
                self.__RSI_Entry(self.rsi_[i], ind[0])
                self.__PsychoEntry(self.psycho_[i], ind[2])
                self.__StochasEntry(self.stochas_[i], ind[4])
                self.__DevEntry(self.devnor_[i], ind[6])
                L_signal = self.__PosiEntry()

                self.__RSI_Entry(self.rsi_[i], ind[1])
                self.__PsychoEntry(self.psycho_[i], ind[3])
                self.__StochasEntry(self.stochas_[i], ind[5])
                self.__DevEntry(self.devnor_[i], ind[7])
                S_signal = self.__PosiEntry()

                if L_signal == 1 and S_signal == -1:
                    #買い売り共に点灯したら評価しない
                    pafo_list = []
                    break
                elif L_signal == 1:
                    L_posi = 1
                    L_price = self.open_[i-1] #次の日の初値で買う
                    continue
                else:
                    continue
            #買いポジを持っていた場合
            else: #L_posi == 1
                #パフォーマンスを計算 -10%以下で損切り ポジションをクローズ
                draw_down_rate = self.PafoRate(L_price, self.close_[i])
                if draw_down_rate <= -0.1:
                    pafo_rate = self.PafoRate(L_price, self.open_[i-1])
                    posi.append(['L', L_price, self.open_[i-1]])
                    L_posi = 0
                    L_price = 0
                #+-1%以下では判定しない
                elif abs(draw_down_rate) <= 0.01:
                    continue
                else: #決済判定
                    #各指標によるショートシグナルを計算
                    self.__RSI_Entry(self.rsi_[i], ind[1])
                    self.__PsychoEntry(self.psycho_[i], ind[3])
                    self.__StochasEntry(self.stochas_[i], ind[5])
                    self.__DevEntry(self.devnor_[i], ind[7])
                    S_signal = self.__PosiEntry()
                    #決済執行
                    if S_signal == -1:
                        L_posi = 0
                        #次の日の初値で売る
                        pafo_rate = self.PafoRate(L_price, self.open_[i-1])
                        posi.append(['L', L_price, self.open_[i-1]])
                        L_price = 0
                    else:
                        continue

            pafo_list.append(pafo_rate)

        #TRADE_NUM日に一回は取引する
        if len(pafo_list) < date_len/TRADE_NUM:
            pafo_list = []

        return pafo_list, posi


    def LongShort(self, ind):
        """
        買いからのドテン売り，空売りからのドテン買い
        return: Paformance and position list.
        """
        date_len = len(self.open_) #データの長さ
        L_posi = 0 #Positionの有無
        S_posi = 0
        pafo_list = [] #損益率
        posi = []
        #時系列順に
        for i in reversed(range(1,date_len)):
            #まずは買いポジと売りポジどちらからスタートするか判断
            if L_posi == 0 and S_posi == 0:

                #買い売り共に点灯することがあれば評価しない
                if ind[0] >= ind[1]:
                    pafo_list = []
                    break
                elif ind[2] >= ind[3]:
                    pafo_list = []
                    break
                elif ind[4] >= ind[5]:
                    pafo_list = []
                    break
                elif ind[6] >= ind[7]:
                    pafo_list = []
                    break
                else:
                    pass

                #各指標によるシグナルを計算
                self.__RSI_Entry(self.rsi_[i], ind[0])
                self.__PsychoEntry(self.psycho_[i], ind[2])
                self.__StochasEntry(self.stochas_[i], ind[4])
                self.__DevEntry(self.devnor_[i], ind[6])
                L_signal = self.__PosiEntry()

                self.__RSI_Entry(self.rsi_[i], ind[1])
                self.__PsychoEntry(self.psycho_[i], ind[3])
                self.__StochasEntry(self.stochas_[i], ind[5])
                self.__DevEntry(self.devnor_[i], ind[7])
                S_signal = self.__PosiEntry()

                if L_signal == 1 and S_signal == -1:
                    #買い売り共に点灯したら評価しない
                    #上の処理のおかげでこうなることはない
                    pafo_list = []
                    break
                elif L_signal == 1:
                    L_posi = 1
                    L_price = self.open_[i-1] #次の日の初値で買う
                    continue
                elif S_signal == -1:
                    S_posi = 1
                    S_price = self.open_[i-1] #次の日の初値で空売り
                    continue
                else:
                    continue
            #買いポジを持っていた場合
            elif L_posi == 1 and S_posi == 0:
                #パフォーマンスを計算 -10%以下で損切り ポジションをクローズ
                draw_down_rate = self.PafoRate(L_price, self.close_[i])
                if draw_down_rate <= -0.1:
                    pafo_rate = self.PafoRate(L_price, self.open_[i-1])
                    posi.append(['L', L_price, self.open_[i-1]])
                    L_posi = 0
                    S_posi = 0
                    L_price = 0
                    S_price = 0
                #+-1%以下では判定しない
                elif abs(draw_down_rate) <= 0.01:
                    continue
                else: #決済＆ドテンの判定
                    #各指標によるショートシグナルを計算
                    self.__RSI_Entry(self.rsi_[i], ind[1])
                    self.__PsychoEntry(self.psycho_[i], ind[3])
                    self.__StochasEntry(self.stochas_[i], ind[5])
                    self.__DevEntry(self.devnor_[i], ind[7])
                    S_signal = self.__PosiEntry()
                    #決済＆ドテンを執行
                    if S_signal == -1:
                        L_posi = 0
                        S_posi = 1
                        #次の日の初値で売る
                        pafo_rate = self.PafoRate(L_price, self.open_[i-1])
                        posi.append(['L', L_price, self.open_[i-1]])
                        #ドテンショート
                        L_price = 0
                        S_price = self.open_[i-1]
                    else:
                        continue
            #売りポジを持っていた場合
            elif L_posi == 0 and S_posi == 1:
                #パフォーマンスを計算 -10%以下で損切り ポジションをクローズ
                #ショートの計算
                draw_down_rate = -self.PafoRate(S_price, self.close_[i])
                if draw_down_rate <= -0.1:
                    pafo_rate = 0 - self.PafoRate(S_price, self.open_[i-1])
                    posi.append(['S', S_price, self.open_[i-1]])
                    L_posi = 0
                    S_posi = 0
                    L_price = 0
                    S_price = 0
                elif abs(draw_down_rate) <= 0.01:
                    continue
                else: #決済＆ドテンの判定
                    #各指標によるロングシグナルを計算
                    self.__RSI_Entry(self.rsi_[i], ind[0])
                    self.__PsychoEntry(self.psycho_[i], ind[2])
                    self.__StochasEntry(self.stochas_[i], ind[4])
                    self.__DevEntry(self.devnor_[i], ind[6])
                    L_signal = self.__PosiEntry()
                    #決済＆ドテンを執行
                    if L_signal == 1:
                        L_posi = 1
                        S_posi = 0
                        #次の日の初値で買い戻す
                        pafo_rate = -self.PafoRate(S_price, self.open_[i-1])
                        posi.append(['S', S_price, self.open_[i-1]])
                        #ドテンロング
                        L_price = self.open_[i-1]
                        S_price = 0
                    else:
                        continue
            #L_posi == 1 and S_posi == 1 となることはない
            else:
                print("Error: Cross order.")

            #ここでパフォーマンスをリストに記録
            pafo_list.append(pafo_rate)

        #TRADE_NUM日に一回は取引する
        if len(pafo_list) < date_len/TRADE_NUM:
            pafo_list = []

        return pafo_list, posi


    #このあたりの逆は張り系指標のシグナル判定は全部一つに纏められそう，まあいいか
    def __RSI_Entry(self, rsi_value, ind_gene):
        """
        rsi_value: RSIの値
        ind_gene: 基準となる値
        個体の遺伝子をrsiデータが下回ったら買いサイン，上回ったら売りサイン
        """
        self.rsi_signal = 0
        if math.isnan(rsi_value) == True:
            pass
        #rsiデータが個体の遺伝子を下回ったら買いサインを点灯
        elif rsi_value <= ind_gene:
            self.rsi_signal = 1
        else:
            self.rsi_signal = -1


    def __PsychoEntry(self, psycho_value, ind_gene):
        """
        psycho_value: Psychologicalの値
        ind_gene: 基準となる値
        個体の遺伝子をpsychoデータが下回ったら買いサイン，上回ったら売りサイン
        """
        self.psycho_signal = 0
        if math.isnan(psycho_value) == True:
            pass
        elif psycho_value <= ind_gene:
            self.psycho_signal = 1
        else:
            self.psycho_signal = -1


    def __StochasEntry(self, stochas_value, ind_gene):
        """
        stochas_value: Stochasticsの値
        ind_gene: 基準となる値
        個体の遺伝子をstochasデータが下回ったら買いサイン，上回ったら売りサイン
        """
        self.stochas_signal = 0
        if math.isnan(stochas_value) == True:
            pass
        elif stochas_value <= ind_gene:
            self.stochas_signal = 1
        else:
            self.stochas_signal = -1


    def __DevEntry(self, devnor_value, ind_gene):
        """
        dev_value: 25Deviation
        ind_gene: 基準となる値
        個体の遺伝子をdevデータが下回ったら買いサイン，上回ったら売りサイン
        """
        self.dev_signal = 0
        if math.isnan(devnor_value) == True:
            pass
        elif devnor_value <= ind_gene:
            self.dev_signal = 1
        else:
            self.dev_signal = -1


    def __PosiEntry(self):
        """
        ポジションのエントリーを判断
        """
        signal = 0
        signal_sum = (self.rsi_signal + self.psycho_signal
                      + self.stochas_signal + self.dev_signal)
        #4つのシグナルのうち3つ以上点灯で判断
        if signal_sum >= 2:
            signal = 1
        elif signal_sum <= -2:
            signal = -1
        else:
            signal = 0

        return signal


    def PafoRate(self, in_price, out_price):
        """
        Calculation paformance rate
        """
        pafo_rate_value = (out_price/in_price)-1
        return pafo_rate_value


if __name__ == '__main__':
    print('Yes!')
