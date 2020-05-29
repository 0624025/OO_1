from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates  # matplotlib 繪圖套件
import argparse
import os


class Count:

    def __init__(self, stock, graph):
        '''初始化'''
        self.stock = stock  # 股票名稱
        self.lower_than_2SD = False  # 是否低於-2SD線
        self.graph = graph  # 決定要不要畫圖
        self.close = []  # 每日收盤價
        self.date = []  # 資料日期
        self._2SD = 0  # -2SD線

        if len(stock) == 4:
            self.get_data()

    def check(self, _2SD):
        '''檢查是否低於-2sd'''
        if _2SD > self.close[-1]:
            self.lower_than_2SD = True
            print("股價低於趨勢線")

    def draw(self, avg, std):
        '''繪圖'''
        # 'TL', '+1SD', '+2SD', '-1SD', '-2SD', 'close'
        data = [self.date, avg, [], [], [], [], self.close]
        p = [0, 0, 0.84, 0.976, 0.158, 0.022]
        '''把五條線的資料全部放進去'''
        for i in range(0, len(avg)):
            for j in range(2, 6):
                data[j].append(round(norm.ppf(p[j], avg[i], std[i]), 2))

        '''判斷是否低於-2SD線'''
        self.check(data[5][-1])

        if self.graph:
            '''五線譜視覺化'''
            plt.plot(data[0], data[6], linewidth=2, color="red")
            plt.plot(data[0], data[1], color="black")
            plt.plot(data[0], data[2], color="black")
            plt.plot(data[0], data[3], color="black")
            plt.plot(data[0], data[4], color="black")
            plt.plot(data[0], data[5], color="black")

            '''圖表標籤'''
            plt.title(self.stock)
            freq = 30  # x軸頻率 (30天)
            plt.xticks(data[0][::freq])
            plt.xticks(fontsize=10, rotation=45)
            plt.show()

    def notation(self):
        '''計算繪圖所需資料'''
        avg = []
        dev = []
        std = []
        count = 1

        '''這邊就是算每日收盤價的線性回歸
           可以想像每天的價錢都是一個點
           然後畫出一條最接近的線(y=mx+b)
           avg = count * slope + intercept '''
        x = np.arange(1, len(self.close)+1).reshape((-1, 1))
        y = self.close
        model = LinearRegression().fit(x, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        '''上面決定 slope, intercept (m,b)是什麼
           這邊算出這條線每個 x 點的 y 值
           或理解成 每一天 的 平均價(五線譜中間那條)'''
        for i in self.close:
            avg.append(np.round(count * slope + intercept, 2))
            dev.append(i - avg[count - 1])
            count += 1

        '''算出標準差'''
        for i in self.close:
            std.append(np.round(np.std(dev), 2))

        self.draw(avg, std)

    def get_data(self):
        '''取得csv資料後計算'''

        '''得到檔案路徑'''
        path = os.getcwd() + "/data/{}.csv".format(self.stock)
        dataset = pd.read_csv(path)  # dataset就是一隻股票全部的資料

        print("最新五天資料")
        print(dataset.head())

        '''取得 date 和 close 這兩項每日的資料並且反轉
           因為原本是由新到舊到我們要先從最老的開始算    '''
        self.date = np.array(dataset["date"])[::-1]
        self.close = np.array(dataset["close"])[::-1]

        '''開始計算'''
        self.notation()


'''______________________________________________________________________'''


def check_stock_exist(stock):
    '''確認 csv 檔是否存在'''
    file_name = '{}/{}.csv'.format('data', stock)
    if os.path.isfile(file_name):
        return True
    else:
        return False


def count_all():
    '''計算所有股票'''
    print("計算全部股票，資料龐大請稍後")
    path = os.getcwd() + "/data/"
    files = listdir(path)

    for f in files:
        stock = f.strip('.csv')
        print(stock)
        count = Count(stock, False)
        # if count.lower_than_2SD:
        #     print(stock)
        del count
    print('全部計算完成!')


def main():
    # get arguments
    parser = argparse.ArgumentParser(description='搜尋股票資訊')
    parser.add_argument('stock', type=str, nargs='*',
                        help='請輸入欲查詢的股票代碼')
    parser.add_argument('-a', '--all', action='store_true',
                        help='計算所有股票')
    args = parser.parse_args()

    if len(args.stock) == 1:
        '''計算單隻股票 (會畫圖)'''
        stock = args.stock[0]  # 傳入股票代碼
        if check_stock_exist(stock):
            print('計算 {} 中'.format(stock))
            count = Count(stock, True)
        else:
            parser.error('查無代碼！')
            return
    elif args.all:
        '''計算全部股票 印出符合條件的 (不畫圖)'''
        count_all()
    else:
        parser.error('格式錯誤！')
        return


if __name__ == '__main__':
    main()
