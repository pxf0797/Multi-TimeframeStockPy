#!usr/bin/env python3
#coding: utf-8
#GetStock.py

from  Ashare import *
import csvfile

class GetStock:
    ''' '''
    __text_args_list = ['pyScript','/','Stock.txt']
    __csv_args_list = ['pyScript','/','test_data.csv']
    #__text = ''
    __csvfile = ''
    __convertSuffix = 'csv'
    __encoding = 'utf-8' #'utf-8' 'gb2312' 'gbk'
    
    __stock_name = ''
    __df = ''
    __header = ''
    
    def __init__(self):
        self.__stock_name = 'sh000001'
        self.__header = 'day,open,high,low,close,volume'
        
        self.__csvfile = csvfile.csvfile(self.__csv_args_list)
        
    #======================================================
    # Basic Operation
    #------------------------------------------------------
    def SetStockName(self,name):
        self.__stock_name = name
    
    def ShowDf(self):
        print(self.__df)
        
    def GetStock(self,frequency,count):
        '''
        frequency: '1m','5m','15m','30m','60m', '1d'=240m, '1w'=1200m, '1M'=7200m
        counts: days or frequency counts
        '''
        self.__df = get_price(self.__stock_name,frequency=frequency,count=count)
        
    def GetStockDate(self,frequency,count,date):
        '''
        frequency: '1m','5m','15m','30m','60m', '1d'=240m, '1w'=1200m, '1M'=7200m
        counts: days or frequency counts
        '''
        self.__df = get_price(self.__stock_name,frequency=frequency,count=count,end_date=date)
        
    #======================================================
    # Save csv Operation
    #------------------------------------------------------
    def SaveStockCsv(self,frequency):
        #self.__csvfile.writeLists(self.__df)
        # saving the DataFrame as a CSV file
        gfg_csv_data = self.__df.to_csv(frequency+'_test_data.csv', index = True)
        print('\nCSV String:\n', gfg_csv_data)

if __name__ == '__main__':
    gs = GetStock()
    gs.SetStockName('sh000001')
    gs.GetStock(frequency='1m',count=2000)
    gs.ShowDf()
    gs.SaveStockCsv(frequency='1m')
    gs.GetStock(frequency='5m',count=2000)
    gs.SaveStockCsv(frequency='5m')
    gs.GetStock(frequency='15m',count=2000)
    gs.SaveStockCsv(frequency='15m')
    gs.GetStock(frequency='60m',count=2000)
    gs.SaveStockCsv(frequency='60m')
    gs.GetStock(frequency='1d',count=1000)
    gs.SaveStockCsv(frequency='1d')
    gs.GetStock(frequency='1w',count=500)
    gs.SaveStockCsv(frequency='1w')
    gs.GetStock(frequency='1M',count=50)
    gs.SaveStockCsv(frequency='1M_')

