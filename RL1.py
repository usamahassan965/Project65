#!/usr/bin/env python
# coding: utf-8

# # 1. Install and Import dependencies


# Gym stuff
import gym
import gym_anytrading
import streamlit as st
import datetime
# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C

# Processing libraries
from gym_anytrading.envs import StocksEnv
from finta import TA
import numpy as np
import seaborn as sns
import pandas as pd
import quantstats as qs
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.graph_objects import Layout
import yfinance as yf
import string
import warnings
warnings.filterwarnings('ignore')
##################################################################################################################

st.title('Stock Trend Prediction')
st.sidebar.header('User input Features')
Load_check = st.sidebar.checkbox('Load data')

##################################### Scraped Data ###############################################################
# t = []
# def scrape_stock_symbols(letter):
#     letter = letter.upper()
#     headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
#     url = 'https://finance.yahoo.com/lookup?s='+letter
#     page = requests.get(url,headers = headers)
#     soup = BeautifulSoup(page.text,'html.parser')
#     odd_rows = soup.find_all('tr')
#     odd_rows.pop(0)
#     for i in range(len(odd_rows)):
#         row = odd_rows[i].find_all('td')
#         t.append(row[0].text.strip())
#     return t

# ticker = []
# for l in string.ascii_lowercase:
#     ticker.extend(scrape_stock_symbols(l))
    
# options = set(ticker)

###Ticker Symbols
options = ['2YY=F','6J=F','6N=F','AA','AAL','AAPL','AFRM','ALLY','AMC','AMD','AMZN','APE','ARKK','ASML','ATMC','ATMCR',
 'ATMCW','AUY','BA', 'BABA','BAC','BB','BBBY','BBIG','BFRG','BFRGW','BHP','BNGO','BOIL','BRK-B','BTC-USD','BTC=F',
 'BX','C','CALM','CHPT','CL=F','COIN','COST','CRWD','CVKD','CVNA','CVS','CWH','CYAD','D','DAL','DFS','DIA',
 'DIDIY','DIS','DKNG','DQ','DVN','DWAC','EDU','EH','EMR','ENPH','EPD','ES=F','ET','ETH-USD','EURUSD=X','F','FCEL',
 'FCX','FFIE','FNMA','FSR','FTCH','FUBO','FWBI','GC=F','GDX','GE','GEHC','GERN','GGAL','GM','GME','GOLD','GOOG',
 'GOOGL','GS','HD','HKD','HOOD','IBM','INMD','INTC','IQ','IWM','JBHT','JBLU','JD','JEPI','JEPQ','JMIA','JNJ',
 'JOBY','JPM','JPY=X','JPYHKD=X','K','KALA','KE=F','KEY','KHC','KMI','KO','KR','KRW=F','KRWHKD=X','KRWUSD=X',
 'KSS','KWEB','LAC','LAZR','LCID','LLY','LMT','LUMN','M','MARA','MCD','META','MO','MPW','MRNA','MRO','MS',
 'MSFT','MU','MULN','NFLX','NG=F','NIO','NKE','NKLA','NOC','NOK','NOK=F','NQ=F','NRGV','NU','NVAX','NVDA',
 'NYCB','O','OJ=F','OKTA','OMF','OPEN','OPK','ORCL','ORMP','OTRK','OXY','OZSC','PAA','PANW','PARA','PBR',
 'PFE','PG','PLTR','PLUG','PNC','PRTY','PXD','PYPL','QARAED=X','QAREUR=X','QCOM','QQQ','QS','QSG','QYLD','RAD',
 'RBLX','RIG','RIGL','RIO','RIOT','RIVN','RKLB','ROKU','RTX','RUM','S','SAVA','SCHD','SCLX','SDI=F','SHEL',
 'SHIB-USD','SHOP','SI','SI=F','SIRI','SLB','SOFI','SOXL','SOXS','SPCE','SPY','SQ','SQM','SQQQ','SRNE',
 'T','TCBP','TLRY','TLT','TN=F','TQQQ','TSLA','TSM','U','UAL','UBER','UNG','UNH','UPS','UPST','UVXY','V','VALE',
 'VERU','VLO','VLTA','VOO','VRM','VRTX','VTI','VTRS','VXRT','VYM','VZ','W','WBA','WBD','WES','WFC','WKHS','WM',
 'WMT','WYNN','X','XAFBRX=X','XCUJPY=X','XCUMYR=X','XCURUB=X','XCUUSD=X','XDRISX=X','XELA','XLE','XM',
 'XOFUSD=X','XOM','XPEV','XRP-USD','YM=F','YPF','ZARAED=X','ZAREUR=X','ZARINR=X','ZARKRW=X','ZARNOK=X',
 'ZARTWD=X','ZARUSD=X','ZF=F','ZIM','ZM','ZN=F','ZOM','ZS','ZT=F','^DJI','^GSPC','^IXIC','^RUT','^TNX','^VIX']


ticker = st.selectbox("Select the Stock ticker",options) 

c1,c2 = st.columns(2)
with c1:
    Start = st.date_input('Enter Starting Date',datetime.date(2019,1,21))
with c2:
    End = st.date_input('Enter Ending Date',datetime.date(2021,2,23))
                        
def load_data(ticker,Start,End):
    df = yf.download(ticker, Start,End)
    df.reset_index(inplace=True)
    return df


# # # 2. Build Environment

# # # 2.1 Add Custom Indicators

# # ## 2.1.3. Calculate SMA, RSI and OBV

# # ## 2.1.4. Create New Environments


def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'Volume','SMA', 'RSI', 'OBV']].to_numpy()[start:end]
    return prices, signal_features


class MyCustomEnv(StocksEnv):
    _process_data = add_signals

# Plot raw data
def plot_raw_data():
    INCREASING_COLOR = '#17BECF'
    DECREASING_COLOR = '#7F7F7F'
    colors = []

    for i in range(len(df.Close)):
        if i != 0:
            if df.Close[i] > df.Close[i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)
    data = [ dict(
    type = 'candlestick',
    open = df.Open,
    high = df.High,
    low = df.Low,
    close = df.Close,
    x = df.Date,
    yaxis = 'y2',
    name = 'OHLC',increasing = dict( line = dict( color = INCREASING_COLOR ) ),
    decreasing = dict( line = dict( color = DECREASING_COLOR ) )) ]

    layout=dict()

    fig = dict( data=data, layout=layout )
    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['xaxis'] = dict( rangeselector = dict( visible = True ) )
    fig['layout']['yaxis'] = dict( domain = [0, 0.2], showticklabels = False )
    fig['layout']['yaxis2'] = dict( domain = [0.2, 0.8] )
    fig['layout']['legend'] = dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' )
    fig['layout']['margin'] = dict( t=40, b=40, r=40, l=40 )
    
    fig['data'].append( dict( x=df.Date, y=df.Volume,                         
                         marker=dict( color=colors ),
                         type='bar', yaxis='y', name='Volume' ) )
    fig.update_layout(
    autosize=False,
    width=800,
    height=800)
    
    st.plotly_chart(fig, config= {'displaylogo': False})

try:
    if Load_check:
        df = load_data(ticker,Start,End)
        df['SMA'] = TA.SMA(df, 12)
        df['RSI'] = TA.RSI(df)
        df['OBV'] = TA.OBV(df)
        df.fillna(0, inplace=True)
        st.subheader('Raw Data')
        st.write(df.head())
        st.header('OHLCV Graph')
        plot_raw_data()
        
        months_train = st.sidebar.slider('Months to train...',min_value=1,max_value=12,value=2,step=1)          # # # 3. Build Environment and Train

        with st.spinner('Please wait while the model is training  for prediction...'):
            train = 30*months_train
            env2 = MyCustomEnv(df=df, window_size=12, frame_bound=(12,train))
            env_maker = lambda: env2
            env = DummyVecEnv([env_maker])
            model = A2C('MlpPolicy', env, verbose=1) 
            model.learn(total_timesteps=5000)
            months = st.sidebar.slider('Months to predict...',min_value=1,max_value=12,value=2,step=1)
        with st.spinner('Please wait while the model is predicting...'):
            start_index = (train-20)
            end_index = start_index + 30*months
            env = MyCustomEnv(df=df, window_size=12, frame_bound=(start_index,end_index))                             # # # 4. Evaluation
            obs = env.reset()
            while True: 
                obs = obs[np.newaxis, ...]
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                if done:
                    print("info", info)
                    break     
            qs.extend_pandas()
            net_worth = pd.Series(env.history['total_profit'], index=df.index[start_index+1:end_index])
            returns = net_worth.pct_change().iloc[1:]
            start = pd.to_datetime(Start)+datetime.timedelta(days=train-18)
            datelist = pd.date_range(start, periods=end_index-start_index-2)
            d = pd.DataFrame(returns,columns=['Returns'])
            d.index = datelist
            st.subheader('Prediction Graph')
            fig,ax = plt.subplots(figsize=(15,6),facecolor='yellow')
            plt.cla()
            env.render_all()
            plt.grid(True)
            plt.title('Forecasting Graph')
            plt.xlabel('Days')
            plt.ylabel('Short Selling (Red) Vs Long Buying (Green)')
            ax.set_facecolor('#1CC4AF')
            st.pyplot(fig)
            st.subheader('Daily Return Plot')
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=d.index,y=d.Returns,mode='markers+lines',name='Returns'))
            fig1.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='blue')
            st.write(fig1)

            st.subheader('Metrics')
            avg_reward = env.history['total_reward']
            avg_profit = env.history['total_profit']
            max_profit = env.max_possible_profit()
            No_Buy = env.history['position'].count(1)
            No_Sell = env.history['position'].count(0)
            st.markdown(f'Our model has earned **{round(sum(avg_reward)/len(avg_reward),2)}** **percent reward** with the **predicted actions of Sell/Buy** and the average profit of **{round(sum(avg_profit)/len(avg_profit),2)}** percent.')
            st.markdown(f'Actions taken : **Buying -- {No_Buy}** ; **Selling -- {No_Sell}**')
            st.markdown(f' Maximum profit achieved While taking action once: **{max_profit}**')
            st.markdown('**Note:** If the average reward is positive and less than 100%, it means the agent has been penalized for making loss in selling actions(average profit < 100%). If the average reward is negative and less than 100%, it means our model has taken buying actions and bought the stocks with the reduced profit. Hence, the best decision taken!') 
except ValueError:
    st.error('Please select the ticker option!',icon='❌')








