import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import yfinance as yf
import csv

# download from yahoo finance for each sector
yf.pdr_override() 

# change input&output data name
year=18    # 15-19  

for i in range(0,9):
    sec=1+i
    df=pd.read_csv('T20'+str(year)+'_Sec'+str(sec)+'.csv')   ## input dataset
    tickers=list(df.ticker)     

    daily_return = web.get_data_yahoo(tickers,start = "2017-12-31",end = "2018-3-31")['Adj Close']

    # save std_return to csv file
    std_return=daily_return.pct_change().std()
    
    
    with open('T20'+str(year)+'_Sec'+str(sec)+'_std.csv', 'w', encoding='utf-8', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['ticker','pc1','pc2','std'])
        for k in range(0,len(df)):
            for j in range(0,len(std_return.index)):
                if df['ticker'][k]==std_return.index.tolist()[j]:
                    csvwriter.writerow([df['ticker'][k],df['pc1'][k],df['pc2'][k],std_return[j]])
                    
    
    
    
###for a single stock
# for single doc
yf.pdr_override() 

df=pd.read_csv('RFactors_2017.csv')   ## input dataset
tickers=list(df.ticker)     

daily_return = web.get_data_yahoo(tickers,start = "2016-12-31",end = "2017-3-31")['Adj Close']

    # save std_return to csv file
std_return=daily_return.pct_change().std()
    
    
with open('RFactors_2017std.csv', 'w', encoding='utf-8', newline='') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['ticker','SIC','pc1','pc2','std'])
    for k in range(0,len(df)):
        for j in range(0,len(std_return.index)):
            if df['ticker'][k]==std_return.index.tolist()[j]:
                csvwriter.writerow([df['ticker'][k],df['SIC'][k],df['pc1'][k],df['pc2'][k],std_return[j]])
