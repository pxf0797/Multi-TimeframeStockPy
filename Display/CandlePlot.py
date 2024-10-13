# coding=utf-8
# inter_candle.py

import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # 你可以尝试不同的后端，如 'TkAgg', 'Agg', 'Qt5Agg', 'GTK3Agg' 等
import matplotlib.pyplot as plt
import mplfinance as mpf
from tafuncs import macd, dema, rsi, bbands, sma

data = pd.read_csv('test_data.csv', index_col=0)
data.index = pd.to_datetime(data.index)

my_color = mpf.make_marketcolors(up='r',
                                 down='g',
                                 edge='inherit',
                                 wick='inherit',
                                 volume='inherit')
my_style = mpf.make_mpf_style(marketcolors=my_color,
                                  figcolor='(0.82, 0.83, 0.85)',
                                  gridcolor='(0.82, 0.83, 0.85)')

title_font = {'fontname': 'pingfang HK',
              'size':     '16',
              'color':    'black',
              'weight':   'bold',
              'va':       'bottom',
              'ha':       'center'}
large_red_font = {'fontname': 'Arial',
                  'size':     '20',
                  'color':    'red',
                  'weight':   'bold',
                  'va':       'bottom'}
large_green_font = {'fontname': 'Arial',
                    'size':     '20',
                    'color':    'green',
                    'weight':   'bold',
                    'va':       'bottom'}
small_red_font = {'fontname': 'Arial',
                  'size':     '12',
                  'color':    'red',
                  'weight':   'bold',
                  'va':       'bottom'}
small_green_font = {'fontname': 'Arial',
                    'size':     '12',
                    'color':    'green',
                    'weight':   'bold',
                    'va':       'bottom'}
normal_label_font = {'fontname': 'pingfang HK',
                     'size':     '12',
                     'color':    'black',
                     'weight':   'normal',
                     'va':       'bottom',
                     'ha':       'right'}
normal_font = {'fontname': 'Arial',
               'size':     '12',
               'color':    'black',
               'weight':   'normal',
               'va':       'bottom',
               'ha':       'left'}

class InterCandle:
    #fig = ''
    __move_len = 50
    __show_len = 100
    __show_min = 60
    __show_max = 800
    __range_part = 5 #move range parts (1/self.__range_part)
    
    def __init__(self, data, my_style):
        self.pressed = False
        self.xpress = None

        # 初始化交互式K线图对象，历史数据作为唯一的参数用于初始化对象
        self.data = data
        self.style = my_style
        # 设置初始化的K线图显示区间起点为0，即显示第0到第99个交易日的数据（前100个数据）
        data_len = len(data)
        #print(data_len)
        start_len = 0
        show_len = self.__show_len
        if (data_len < show_len):
            #start_len = 0
            show_len = data_len
        else:
            start_len = (data_len-show_len-1)
            #show_len = self.__show_len
        self.idx_start = start_len
        self.idx_range = show_len
        # 设置ax1图表中显示的均线类型
        self.avg_type = 'ma'
        self.indicator = 'macd'
        
        # 初始化figure对象，在figure上建立三个Axes对象并分别设置好它们的位置和基本属性
        self.fig = mpf.figure(style=my_style, figsize=(12, 8), facecolor=(0.82, 0.83, 0.85))
        fig = self.fig
        self.ax1 = fig.add_axes([0.08, 0.25, 0.88, 0.60])
        self.ax2 = fig.add_axes([0.08, 0.15, 0.88, 0.10], sharex=self.ax1)
        self.ax2.set_ylabel('volume')
        self.ax3 = fig.add_axes([0.08, 0.05, 0.88, 0.10], sharex=self.ax1)
        self.ax3.set_ylabel('macd')
        # 初始化figure对象，在figure上预先放置文本并设置格式，文本内容根据需要显示的数据实时更新
        self.t1 = fig.text(0.50, 0.94, '513100.SH - 纳斯达克指数ETF基金', **title_font)
        self.t2 = fig.text(0.12, 0.90, '开/收: ', **normal_label_font)
        self.t3 = fig.text(0.14, 0.89, f'', **large_red_font)
        self.t4 = fig.text(0.14, 0.86, f'', **small_red_font)
        self.t5 = fig.text(0.22, 0.86, f'', **small_red_font)
        self.t6 = fig.text(0.12, 0.86, f'', **normal_label_font)
        self.t7 = fig.text(0.40, 0.90, '高: ', **normal_label_font)
        self.t8 = fig.text(0.40, 0.90, f'', **small_red_font)
        self.t9 = fig.text(0.40, 0.86, '低: ', **normal_label_font)
        self.t10 = fig.text(0.40, 0.86, f'', **small_green_font)
        self.t11 = fig.text(0.55, 0.90, '量(万手): ', **normal_label_font)
        self.t12 = fig.text(0.55, 0.90, f'', **normal_font)
        self.t13 = fig.text(0.55, 0.86, '额(亿元): ', **normal_label_font)
        self.t14 = fig.text(0.55, 0.86, f'', **normal_font)
        self.t15 = fig.text(0.70, 0.90, '涨停: ', **normal_label_font)
        self.t16 = fig.text(0.70, 0.90, f'', **small_red_font)
        self.t17 = fig.text(0.70, 0.86, '跌停: ', **normal_label_font)
        self.t18 = fig.text(0.70, 0.86, f'', **small_green_font)
        self.t19 = fig.text(0.85, 0.90, '均价: ', **normal_label_font)
        self.t20 = fig.text(0.85, 0.90, f'', **normal_font)
        self.t21 = fig.text(0.85, 0.86, '昨收: ', **normal_label_font)
        self.t22 = fig.text(0.85, 0.86, f'', **normal_font)

        fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        #show plot
        reflesh_index = (self.idx_start+self.idx_range)
        self.refresh_texts(self.data.iloc[reflesh_index])
        self.refresh_plot(self.idx_start, self.idx_range)

    def refresh_plot(self, idx_start, idx_range):
        """ 根据最新的参数，重新绘制整个图表
        """
        all_data = self.data
        plot_data = all_data.iloc[idx_start: idx_start + idx_range]

        ap = []
        # 添加K线图重叠均线，根据均线类型添加移动均线或布林带线
        if self.avg_type == 'ma':
            ap.append(mpf.make_addplot(plot_data[['MA5', 'MA10', 'MA20', 'MA60']], ax=self.ax1))
        elif self.avg_type == 'bb':
            ap.append(mpf.make_addplot(plot_data[['bb-u', 'bb-m', 'bb-l']], ax=self.ax1))
        # 添加指标，根据指标类型添加MACD或RSI或DEMA
        if self.indicator == 'macd':
            ap.append(mpf.make_addplot(plot_data[['macd-m', 'macd-s']], ylabel='macd', ax=self.ax3))
            bar_r = np.where(plot_data['macd-h'] > 0, plot_data['macd-h'], 0)
            bar_g = np.where(plot_data['macd-h'] <= 0, plot_data['macd-h'], 0)
            ap.append(mpf.make_addplot(bar_r, type='bar', color='red', ax=self.ax3))
            ap.append(mpf.make_addplot(bar_g, type='bar', color='green', ax=self.ax3))
        elif self.indicator == 'rsi':
            ap.append(mpf.make_addplot([75] * len(plot_data), color=(0.75, 0.6, 0.6), ax=self.ax3))
            ap.append(mpf.make_addplot([30] * len(plot_data), color=(0.6, 0.75, 0.6), ax=self.ax3))
            ap.append(mpf.make_addplot(plot_data['rsi'], ylabel='rsi', ax=self.ax3))
        else:  # indicator == 'dema'
            ap.append(mpf.make_addplot(plot_data['dema'], ylabel='dema', ax=self.ax3))
        # 绘制图表
        mpf.plot(plot_data,
                 ax=self.ax1,
                 volume=self.ax2,
                 addplot=ap,
                 type='candle',
                 style=self.style,
                 datetime_format='%Y-%m',
                 xrotation=0)
        self.fig.show()
        plt.show()

    def refresh_texts(self, display_data):
        """ 更新K线图上的价格文本
        """
        # display_data是一个交易日内的所有数据，将这些数据分别填入figure对象上的文本中
        self.t3.set_text(f'{np.round(display_data["open"], 3)} / {np.round(display_data["close"], 3)}')
        self.t4.set_text(f'{np.round(display_data["change"], 3)}')
        self.t5.set_text(f'[{np.round(display_data["pct_change"], 3)}%]')
        self.t6.set_text(f'{display_data.name.date()}')
        self.t8.set_text(f'{np.round(display_data["high"], 3)}')
        self.t10.set_text(f'{np.round(display_data["low"], 3)}')
        self.t12.set_text(f'{np.round(display_data["volume"] / 10000, 3)}')
        self.t14.set_text(f'{display_data["value"]}')
        self.t16.set_text(f'{np.round(display_data["upper_lim"], 3)}')
        self.t18.set_text(f'{np.round(display_data["lower_lim"], 3)}')
        self.t20.set_text(f'{np.round(display_data["average"], 3)}')
        self.t22.set_text(f'{np.round(display_data["last_close"], 3)}')
        # 根据本交易日的价格变动值确定开盘价、收盘价的显示颜色
        if display_data['change'] > 0:  # 如果今日变动额大于0，即今天价格高于昨天，今天价格显示为红色
        #if display_data['close'] > display_data['open']:
            close_number_color = 'red'
        elif display_data['change'] < 0:  # 如果今日变动额小于0，即今天价格低于昨天，今天价格显示为绿色
        #elif display_data['close'] < display_data['open']:
            close_number_color = 'green'
        else:
            close_number_color = 'black'
        self.t3.set_color(close_number_color)
        self.t4.set_color(close_number_color)
        self.t5.set_color(close_number_color)

    def on_press(self, event):
        if not event.inaxes == self.ax1:
            return
        if event.button != 1:
            return
        self.pressed = True
        self.xpress = event.xdata

        # 切换当前ma类型, 在ma、bb、none之间循环
        if event.inaxes == self.ax1 and event.dblclick == 1:
            if self.avg_type == 'ma':
                self.avg_type = 'bb'
            elif self.avg_type == 'bb':
                self.avg_type = 'none'
            else:
                self.avg_type = 'ma'
        # 切换当前indicator类型，在macd/dma/rsi/kdj之间循环
        if event.inaxes == self.ax3 and event.dblclick == 1:
            if self.indicator == 'macd':
                self.indicator = 'dma'
            elif self.indicator == 'dma':
                self.indicator = 'rsi'
            elif self.indicator == 'rsi':
                self.indicator = 'kdj'
            else:
                self.indicator = 'macd'

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.refresh_plot(self.idx_start, self.idx_range)

    def on_release(self, event):
        self.pressed = False
        dx = int(event.xdata - self.xpress)
        self.idx_start -= dx
        if self.idx_start <= 0:
            self.idx_start = 0
        if self.idx_start >= len(self.data) - self.__move_len:
            self.idx_start = len(self.data) - self.__move_len

    def on_motion(self, event):
        if not self.pressed:
            return
        if not event.inaxes == self.ax1:
            return
        dx = int(event.xdata - self.xpress)
        new_start = self.idx_start - dx
        # 设定平移的左右界限，如果平移后超出界限，则不再平移
        if new_start <= 0:
            new_start = 0
        if new_start >= len(self.data) - self.__move_len:
            new_start = len(self.data) - self.__move_len
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        #self.refresh_texts(self.data.iloc[new_start])
        reflesh_index = (new_start+self.idx_range)
        self.refresh_texts(self.data.iloc[reflesh_index])
        self.refresh_plot(new_start, self.idx_range)

    def on_scroll(self, event):
        # 仅当鼠标滚轮在axes1范围内滚动时起作用
        if event.inaxes != self.ax1:
            return
        if event.button == 'down':
            # 缩小20%显示范围
            scale_factor = 0.8
        if event.button == 'up':
            # 放大20%显示范围
            scale_factor = 1.2
        # 设置K线的显示范围大小
        self.idx_range = int(self.idx_range * scale_factor)
        # 限定可以显示的K线图的范围，最少不能少于30个交易日，最大不能超过当前位置与
        # K线数据总长度的差
        data_length = len(self.data)
        if self.idx_range >= data_length - self.idx_start:
            self.idx_range = data_length - self.idx_start
        if self.idx_range <= 30:
            self.idx_range = 30 
        # 更新图表（注意因为多了一个参数idx_range，refresh_plot函数也有所改动）
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        #self.refresh_texts(self.data.iloc[self.idx_start])
        reflesh_index = (self.idx_start+self.idx_range)
        self.refresh_texts(self.data.iloc[reflesh_index])
        self.refresh_plot(self.idx_start, self.idx_range)
        
    # 键盘按下处理
    def on_key_press(self, event):
        data_length = len(self.data)
        if event.key == 'a':  # avg_type, 在ma,bb,none之间循环
            if self.avg_type == 'ma':
                self.avg_type = 'bb'
            elif self.avg_type == 'bb':
                self.avg_type = 'none'
            elif self.avg_type == 'none':
                self.avg_type = 'ma'
        elif event.key == 'up':  # 向上，看仔细1倍
            if self.idx_range > self.__show_min:
                self.idx_range = int(self.idx_range / 2)
        elif event.key == 'down':  # 向下，看多1倍标的
            if self.idx_range <= self.__show_max:
                self.idx_range = self.idx_range * 2
        elif event.key == 'left':  
            #move half range
            half_range = int(self.idx_range / self.__range_part)
            if self.idx_start > half_range:
                self.idx_start = self.idx_start - half_range
            else:
                self.idx_start = 0
        elif event.key == 'right':
            #move half range
            half_range = int(self.idx_range / self.__range_part)
            if self.idx_start < data_length - half_range:
                self.idx_start = self.idx_start + half_range
            else:
                self.idx_start = (data_length-self.idx_range-1)
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        #self.refresh_texts(self.data.iloc[self.idx_start])
        reflesh_index = (self.idx_start+self.idx_range)
        self.refresh_texts(self.data.iloc[reflesh_index])
        self.refresh_plot(self.idx_start, self.idx_range)


def _add_indicators(data, mav=None, bb_par=None, macd_par=None, rsi_par=None, dema_par=None, **kwargs):
    """ data是一只股票的历史K线数据，包括O/H/L/C/V五组数据或者O/H/L/C四组数据
        并根据这些数据生成以下数据，加入到data中：

        - Moving Average
        - change and percent change
        - average
        - last close
        - Bband
        - macd
        - kdj
        - dma
        - rsi

    Parameters
    ----------
    data : DataFrame
        一只股票的历史K线数据，包括O/H/L/C/V五组数据或者O/H/L/C四组数据

    Returns
    -------
    tuple: (data, parameter_string)
        (pd.DataFrame, str) 添加指标的价格数据表，所有指标的参数字符串，以"/"分隔
    """
    if mav is None:
        mav = (5, 10, 20)
    # 其他indicator的parameter使用默认值
    if dema_par is None:
        dema_par = (30,)
    if macd_par is None:
        macd_par = (9, 12, 26)
    if rsi_par is None:
        rsi_par = (14,)
    if bb_par is None:
        bb_par = (20, 2, 2, 0)
    # 在DataFrame中增加均线信息,并先删除已有的均线：
    assert isinstance(mav, (list, tuple))
    assert all(isinstance(item, int) for item in mav)
    mav_to_drop = [col for col in data.columns if col[:2] == 'MA']
    if len(mav_to_drop) > 0:
        data.drop(columns=mav_to_drop, inplace=True)
    # 排除close收盘价中的nan值：
    close = data.close.iloc[np.where(~np.isnan(data.close))]

    for value in mav:  # 需要处理数据中的nan值，否则会输出全nan值
        data['MA' + str(value)] = sma(close, timeperiod=value)  # 以后还可以加上不同的ma_type
    data['change'] = np.round(close - close.shift(1), 3)
    data['pct_change'] = np.round(data['change'] / close * 100, 2)
    data['value'] = np.round(data['close'] * data['volume'] / 1000000, 2)
    data['upper_lim'] = np.round(data['close'] * 1.1, 3)
    data['lower_lim'] = np.round(data['close'] * 0.9, 3)
    data['last_close'] = data['close'].shift(1)
    data['average'] = data[['open', 'close', 'high', 'low']].mean(axis=1)
    data['volrate'] = data['volume']
    # 添加不同的indicator
    data['dema'] = dema(close, *dema_par)
    data['macd-m'], data['macd-s'], data['macd-h'] = macd(close, *macd_par)
    #try:
    #    data['rsi'] = rsi(close, *rsi_par)
    #    data['bb-u'], data['bb-m'], data['bb-l'] = bbands(close, *bb_par)
    #except Exception as e:
    #    import warnings
    #    warnings.warn(f'Failed to calculate indicators RSI and BBANDS, TA-lib is needed, please install TA-lib!. {e}')
    #    data['rsi'] = np.nan
    #    data['bb-u'] = np.nan

    parameter_string = f'{mav}/{bb_par}/{macd_par}/{rsi_par}/{dema_par}'

    return data, parameter_string


if __name__ == '__main__':
    mav = [5, 10, 20, 60]
    data, parameters = _add_indicators(data,
                                        mav=mav,
                                        indicator='macd',
                                        indicator_par=(5, 10, 5))
    candle = InterCandle(data, my_style)
    
    #candle.refresh_texts(data.iloc[249])
    #candle.refresh_texts(data.iloc[-1])
    #candle.refresh_plot(150, 100)
    #candle.refresh_plot(start_len, show_len)
    