#!/usr/bin/env python
# coding: utf-8

# # 0. Install and Import dependencies

# In[1]:


#!pip install tensorflow-gpu==1.15.0 tensorflow==1.15.0 stable-baselines gym-anytrading gym


# In[1]:


# Gym stuff
import gym
import gym_anytrading
import streamlit as st
# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C

# Processing libraries
from gym_anytrading.envs import StocksEnv
from plotly import graph_objs as go
from finta import TA
import numpy as np
import seaborn as sns
import pandas as pd
import quantstats as qs
from matplotlib import pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
##################################################################################################################

st.title('Stock Trend Prediction')

ticker = st.text_input('Enter Stock Ticker','AAPL')

Start = st.text_input('Enter starting date i.e, yyyy-mm-dd','2019-01-01')
End = st.text_input('Enter ending date i.e, yyyy-mm-dd','2020-01-01')

def load_data(ticker,Start,End):
    df = yf.download(ticker, Start,End)
    df.reset_index(inplace=True)
    return df

data_load_state = st.text('Loading data...')
df = load_data(ticker,Start,End)
data_load_state.text('Loading data... done!')

st.subheader('Raw Data')
st.write(df.head())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df.index, y=df['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()
# In[3]:
st.sidebar.header('User Input Features')

df['SMA'] = TA.SMA(df, 12)
df['RSI'] = TA.RSI(df)
df['OBV'] = TA.OBV(df)
df.fillna(0, inplace=True)


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
    
env2 = MyCustomEnv(df=df, window_size=12, frame_bound=(12,50))


# # # 3. Build Environment and Train

env_maker = lambda: env2
env = DummyVecEnv([env_maker])
model = A2C('MlpPolicy', env, verbose=1) 
model.learn(total_timesteps=1000)


# # # 4. Evaluation

# # In[8]:


start_index = 80
end_index = 150
env = MyCustomEnv(df=df, window_size=12, frame_bound=(start_index,end_index))
obs = env.reset()
while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

st.subheader('Prediction Graph')
fig = plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
st.pyplot(fig)

# # In[10]:


qs.extend_pandas()

net_worth = pd.Series(env.history['total_profit'], index=df.index[start_index+1:end_index])
returns = net_worth.pct_change().iloc[1:]

result = pd.DataFrame(returns)

st.subheader('Plot_Daily_Returns')
st.line_chart(result)

st.subheader('Plot_Monthly_Returns')
# profit_color = [{p<0: 'red', 0<=p<=1: 'orange'}[True] for p in df.monthly_returns().T['2019']]
# plt.figure(figsize=(12,6))
# plt.bar(x=df1.index,height=df1['2019'],color=profit_color)




