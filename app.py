import streamlit as st
import pandas as pd
from FinMind.data import DataLoader

# -----------------------------
# FinMind Token（預設寫程式裡）
API_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0wNy0yMyAwOTo0NzoyNyIsInVzZXJfaWQiOiJtOTYwNjI4NiIsImlwIjoiNjAuMjUwLjg5LjkxIn0.XlB7BmkZTaerzA_Wm4PB_MzfbL7i8-nqoQjj2UUV3KM'

# 初始化 API
api = DataLoader()
api.login_by_token(api_token=API_TOKEN)

# -----------------------------
# Streamlit UI
st.title("股票策略回測")

stock_input = st.text_input("股票代號/名稱", value="2330")
start_date = st.date_input("回測開始日", pd.to_datetime("2025-01-01"))
end_date = st.date_input("回測結束日")

st.subheader("選擇策略與參數")
use_ma = st.checkbox("均線交叉", value=True)
if use_ma:
    ma_short = st.number_input("短期均線天數", value=5, step=1)
    ma_long = st.number_input("長期均線天數", value=20, step=1)

use_kd = st.checkbox("KD交叉")
if use_kd:
    kd_period = st.number_input("KD週期", value=9, step=1)
    kd_threshold = st.number_input("KD差異閾值", value=2, step=0.1)

use_macd = st.checkbox("MACD交叉")
if use_macd:
    macd_fast = st.number_input("MACD EMA快速", value=12, step=1)
    macd_slow = st.number_input("MACD EMA慢速", value=26, step=1)
    macd_signal = st.number_input("MACD DEA週期", value=9, step=1)

use_rsi = st.checkbox("RSI超買超賣")
if use_rsi:
    rsi_period = st.number_input("RSI週期", value=14, step=1)

run_bt = st.button("執行回測")

# -----------------------------
# 回測邏輯
if run_bt:
    st.write("回測中...")
    user_input = stock_input.strip()

    # 取得股票資訊
    df_stock_info = api.taiwan_stock_info()
    if user_input.isdigit():
        stock_id = user_input
        stock_row = df_stock_info[df_stock_info['stock_id'] == stock_id]
    else:
        stock_row = df_stock_info[df_stock_info['stock_name'].str.contains(user_input)]
        if stock_row.empty:
            st.error("找不到對應股票名稱")
            st.stop()
        stock_id = stock_row['stock_id'].iloc[0]
    stock_name = stock_row['stock_name'].iloc[0]

    # 取得股價資料
    start_date_api = (pd.to_datetime(start_date) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date_api)
    if df.empty:
        st.error("查無股價資料")
        st.stop()
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df[df['close'] > 0].copy()

    results = []

    # -----------------------------
    # 均線交叉
    if use_ma:
        df['短期均價'] = df['close'].rolling(ma_short, min_periods=1).mean()
        df['長期均價'] = df['close'].rolling(ma_long, min_periods=1).mean()
        df['策略'] = 0
        df.loc[(df['短期均價'] > df['長期均價']) & (df['短期均價'].shift(1) <= df['長期均價'].shift(1)), '策略'] = 1
        df.loc[(df['短期均價'] < df['長期均價']) & (df['短期均價'].shift(1) >= df['長期均價'].shift(1)), '策略'] = -1
        df_bt = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        signals = df_bt[df_bt['策略'] != 0]
        if signals.empty:
            ma_return = 0
        else:
            ret = []
            prices = signals['close'].tolist()
            sigs = signals['策略'].tolist()
            for i in range(len(sigs)):
                entry = prices[i]
                if i < len(sigs)-1:
                    exit_price = prices[i+1]
                else:
                    exit_price = df_bt['close'].iloc[-1]
                r = (exit_price/entry-1)*100 if sigs[i]==1 else (entry/exit_price-1)*100
                ret.append(r)
            ma_return = (1+pd.Series([x/100 for x in ret])).prod()-1
        results.append(("均線交叉", ma_return))

    # -----------------------------
    # KD交叉
    if use_kd:
        low_min = df['min'].rolling(kd_period, min_periods=1).min()
        high_max = df['max'].rolling(kd_period, min_periods=1).max()
        df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(alpha=1/3).mean()
        df['D'] = df['K'].ewm(alpha=1/3).mean()
        diff = df['K'] - df['D']
        diff_prev = diff.shift(1)
        df['策略'] = 0
        df.loc[(diff >= kd_threshold) & (diff_prev < kd_threshold), '策略'] = 1
        df.loc[(diff <= -kd_threshold) & (diff_prev > -kd_threshold), '策略'] = -1
        df_bt = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        signals = df_bt[df_bt['策略'] != 0]
        if signals.empty:
            kd_return = 0
        else:
            ret = []
            prices = signals['close'].tolist()
            sigs = signals['策略'].tolist()
            for i in range(len(sigs)):
                entry = prices[i]
                if i < len(sigs)-1:
                    exit_price = prices[i+1]
                else:
                    exit_price = df_bt['close'].iloc[-1]
                r = (exit_price/entry-1)*100 if sigs[i]==1 else (entry/exit_price-1)*100
                ret.append(r)
            kd_return = (1+pd.Series([x/100 for x in ret])).prod()-1
        results.append(("KD交叉", kd_return))

    # -----------------------------
    # MACD交叉
    if use_macd:
        df['EMA_fast'] = df['close'].ewm(span=macd_fast, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=macd_slow, adjust=False).mean()
        df['DIF'] = df['EMA_fast'] - df['EMA_slow']
        df['DEA'] = df['DIF'].ewm(span=macd_signal, adjust=False).mean()
        df['策略'] = 0
        df.loc[(df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1)), '策略'] = 1
        df.loc[(df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1)), '策略'] = -1
        df_bt = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        signals = df_bt[df_bt['策略'] != 0]
        if signals.empty:
            macd_return = 0
        else:
            ret = []
            prices = signals['close'].tolist()
            sigs = signals['策略'].tolist()
            for i in range(len(sigs)):
                entry = prices[i]
                if i < len(sigs)-1:
                    exit_price = prices[i+1]
                else:
                    exit_price = df_bt['close'].iloc[-1]
                r = (exit_price/entry-1)*100 if sigs[i]==1 else (entry/exit_price-1)*100
                ret.append(r)
            macd_return = (1+pd.Series([x/100 for x in ret])).prod()-1
        results.append(("MACD交叉", macd_return))

    # -----------------------------
    # RSI超買超賣
    if use_rsi:
        delta = df['close'].diff()
        gain = delta.where(delta>0,0)
        loss = -delta.where(delta<0,0)
        avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
        rs = avg_gain/avg_loss
        df['RSI'] = 100 - (100/(1+rs))
        df.loc[(avg_loss==0)&(avg_gain>0),'RSI']=100
        df.loc[(avg_gain==0)&(avg_loss>0),'RSI']=0
        df.loc[(avg_gain==0)&(avg_loss==0),'RSI']=50
        df['策略']=0
        df.loc[(df['RSI'].shift(1)>=30)&(df['RSI']<30),'策略']=1
        df.loc[(df['RSI'].shift(1)<=70)&(df['RSI']>70),'策略']=-1
        df_bt = df[(df['date']>=start_date)&(df['date']<=end_date)].copy()
        signals = df_bt[df_bt['策略']!=0]
        if signals.empty:
            rsi_return = 0
        else:
            ret=[]
            prices=signals['close'].tolist()
            sigs=signals['策略'].tolist()
            for i in range(len(sigs)):
                entry = prices[i]
                if i<len(sigs)-1:
                    exit_price = prices[i+1]
                else:
                    exit_price = df_bt['close'].iloc[-1]
                r = (exit_price/entry-1)*100 if sigs[i]==1 else (entry/exit_price-1)*100
                ret.append(r)
            rsi_return = (1+pd.Series([x/100 for x in ret])).prod()-1
        results.append(("RSI超買超賣", rsi_return))

    # -----------------------------
    # 顯示累積報酬率
    st.subheader(f"{stock_id} {stock_name} 回測區間: {start_date} ~ {end_date}")
    for name, val in results:
        color = "red" if val>0 else "green"
        st.markdown(f"<span style='color:{color}'>{name} 累積報酬率 {val*100:.1f}%</span>", unsafe_allow_html=True)

    # Buy & Hold
    buy_hold = (df_bt['close'].iloc[-1]/df_bt['close'].iloc[0]-1)
    color_bh = "red" if buy_hold>0 else "green"
    st.markdown(f"<span style='color:{color_bh}'>Buy & Hold 累積報酬率 {buy_hold*100:.1f}%</span>", unsafe_allow_html=True)
