import streamlit as st
import pandas as pd
from FinMind.data import DataLoader
from datetime import datetime

# -----------------------------
# 初始化 FinMind API
api = DataLoader()
api.login_by_token(api_token='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0wNy0yMyAwOTo0NzoyNyIsInVzZXJfaWQiOiJtOTYwNjI4NiIsImlwIjoiNjAuMjUwLjg5LjkxIn0.XlB7BmkZTaerzA_Wm4PB_MzfbL7i8-nqoQjj2UUV3KM')

# -----------------------------
# UI 元件
st.title("台股策略回測系統")

stock_input = st.text_input("股票代號/名稱", value='2330')
date_input = st.date_input("回測開始日", value=pd.to_datetime('2025-01-01'))
date_output = st.date_input("回測結束日", value=None)
strategy_select = st.radio("策略", ['均線交叉', 'KD交叉', 'MACD交叉', 'RSI超買超賣'])

# 動態策略參數
param_values = {}
if strategy_select == '均線交叉':
    param_values['short_ma'] = st.number_input("短天數", value=5, min_value=1)
    param_values['long_ma'] = st.number_input("長天數", value=20, min_value=1)
elif strategy_select == 'KD交叉':
    param_values['kd_period'] = st.number_input("週期", value=9, min_value=1)
    param_values['kd_threshold'] = st.number_input("閾值", value=2)
elif strategy_select == 'MACD交叉':
    param_values['fast'] = st.number_input("短期EMA", value=12, min_value=1)
    param_values['slow'] = st.number_input("長期EMA", value=26, min_value=1)
    param_values['signal'] = st.number_input("訊號EMA", value=9, min_value=1)
elif strategy_select == 'RSI超買超賣':
    param_values['period'] = st.number_input("週期", value=14, min_value=1)

run_button = st.button("執行回測")

# -----------------------------
# 執行回測
def run_backtest():
    user_input = stock_input.strip()
    user_start_date = pd.to_datetime(date_input).date()

    # 取得股票資訊
    df_stock_info = api.taiwan_stock_info()
    if user_input.isdigit():
        stock_id = user_input
        stock_row = df_stock_info[df_stock_info['stock_id'] == stock_id]
    else:
        stock_row = df_stock_info[df_stock_info['stock_name'].str.contains(user_input)]
        if stock_row.empty:
            st.error("找不到對應股票名稱")
            return
        stock_id = stock_row['stock_id'].iloc[0]
    stock_name = stock_row['stock_name'].iloc[0]

    # 取得股價資料
    start_date_api = (user_start_date - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date_api)
    if df.empty:
        st.error("查無股價資料")
        return
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df[df['close'] > 0].copy()

    # -----------------------------
    # 計算技術指標
    if strategy_select == '均線交叉':
        short_ma = param_values['short_ma']
        long_ma = param_values['long_ma']
        df['短期均價'] = df['close'].rolling(short_ma, min_periods=1).mean()
        df['長期均價'] = df['close'].rolling(long_ma, min_periods=1).mean()
    elif strategy_select == 'KD交叉':
        kd_period = param_values['kd_period']
        kd_threshold = param_values['kd_threshold']
        low_min = df['min'].rolling(kd_period, min_periods=1).min()
        high_max = df['max'].rolling(kd_period, min_periods=1).max()
        df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(alpha=1/3).mean()
        df['D'] = df['K'].ewm(alpha=1/3).mean()
        diff = df['K'] - df['D']
        diff_prev = diff.shift(1)
    elif strategy_select == 'MACD交叉':
        fast = param_values['fast']
        slow = param_values['slow']
        signal = param_values['signal']
        df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
        df['DIF'] = df['EMA_fast'] - df['EMA_slow']
        df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    elif strategy_select == 'RSI超買超賣':
        period = param_values['period']
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df.loc[(avg_loss == 0) & (avg_gain > 0), 'RSI'] = 100
        df.loc[(avg_gain == 0) & (avg_loss > 0), 'RSI'] = 0
        df.loc[(avg_gain == 0) & (avg_loss == 0), 'RSI'] = 50

    # -----------------------------
    # 計算策略訊號
    df['策略'] = 0
    df['狀態'] = ''
    if strategy_select == '均線交叉':
        df.loc[(df['短期均價'] > df['長期均價']) & (df['短期均價'].shift(1) <= df['長期均價'].shift(1)), ['策略','狀態']] = [1,'短期均線黃金交叉']
        df.loc[(df['短期均價'] < df['長期均價']) & (df['短期均價'].shift(1) >= df['長期均價'].shift(1)), ['策略','狀態']] = [-1,'短期均線死亡交叉']
    elif strategy_select == 'KD交叉':
        buy_kd = (diff >= kd_threshold) & (diff_prev < kd_threshold)
        sell_kd = (diff <= -kd_threshold) & (diff_prev > -kd_threshold)
        df.loc[buy_kd, ['策略','狀態']] = [1,'K黃金交叉D(差>=閾值)']
        df.loc[sell_kd, ['策略','狀態']] = [-1,'K死亡交叉D(差<=-閾值)']
    elif strategy_select == 'MACD交叉':
        df.loc[(df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1)), ['策略','狀態']] = [1, 'MACD黃金交叉']
        df.loc[(df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1)), ['策略','狀態']] = [-1, 'MACD死亡交叉']
    elif strategy_select == 'RSI超買超賣':
        df.loc[(df['RSI'].shift(1) >= 30) & (df['RSI'] < 30), ['策略','狀態']] = [1,'RSI低檔超賣']
        df.loc[(df['RSI'].shift(1) <= 70) & (df['RSI'] > 70), ['策略','狀態']] = [-1,'RSI高檔超買']

    # -----------------------------
    # 過濾回測區間
    end_date_val = pd.to_datetime(date_output).date() if date_output else df['date'].iloc[-1]
    df_backtest = df[(df['date'] >= user_start_date) & (df['date'] <= end_date_val)].copy()
    if df_backtest.empty:
        st.warning("回測區間無資料")
        return

    # -----------------------------
    # 交易紀錄
    raw_signals = df_backtest[df_backtest['策略'] != 0][['date','策略','close','狀態']].reset_index(drop=True)
    mask = (raw_signals['策略'] != raw_signals['策略'].shift(1))
    trade_records = raw_signals[mask].reset_index(drop=True)
    if trade_records.empty:
        st.info("回測期間沒有策略訊號")
        return
    trade_records['策略'] = [
        ("買進" if sig==1 else "放空") if i==0 else ("回補+買進" if sig==1 else "賣出+放空")
        for i, sig in enumerate(trade_records['策略'])
    ]

    # 計算報酬率
    strategy_returns, date_range, close_range = [], [], []
    for i in range(len(trade_records)):
        entry_date = trade_records['date'].iloc[i]
        entry_price = trade_records['close'].iloc[i]
        sig = trade_records['策略'].iloc[i]
        if i < len(trade_records)-1:
            exit_date = trade_records['date'].iloc[i+1]
            exit_price = trade_records['close'].iloc[i+1]
        else:
            exit_date = df_backtest['date'].iloc[-1]
            exit_price = df_backtest['close'].iloc[-1]
        ret = (exit_price/entry_price-1)*100 if "買進" in sig else (entry_price/exit_price-1)*100
        strategy_returns.append(ret)
        date_range.append(f"{entry_date}~{exit_date}")
        close_range.append(f"{entry_price:.2f}~{exit_price:.2f}")

    trade_records['策略區間'] = date_range
    trade_records['收盤價區間'] = close_range
    trade_records['報酬率'] = [f"{x:.1f}%" for x in strategy_returns]
    trade_records = trade_records[['策略區間','狀態','策略','報酬率','收盤價區間']]

    # -----------------------------
    # 加入序號欄位
    trade_records.insert(0, '序號', range(1, len(trade_records)+1))

    # -----------------------------
    # 累積報酬率
    strategy_cum_return = (1 + pd.Series([x/100 for x in strategy_returns])).prod() - 1
    buy_hold_return = (df_backtest['close'].iloc[-1] / df_backtest['close'].iloc[0]) - 1

    st.markdown(
        f"**策略累計報酬率:** <span style='color:{'red' if strategy_cum_return>0 else 'green'}'>{strategy_cum_return*100:.1f}%</span> | "
        f"**Buy & Hold 累計報酬率:** <span style='color:{'red' if buy_hold_return>0 else 'green'}'>{buy_hold_return*100:.1f}%</span>",
        unsafe_allow_html=True
    )

    # -----------------------------
    # 顯示交易紀錄表格（無索引）
    def color_profit(val):
        if isinstance(val,str) and '%' in val:
            num = float(val.strip('%'))
            return 'color:red' if num>0 else 'color:green' if num<0 else ''
        return ''

    st.dataframe(
        trade_records.style
        .map(color_profit, subset=['報酬率'])
        .set_properties(**{'text-align':'right'})
        .hide(axis="index")   # 隱藏索引
    )

# -----------------------------
# 按鈕觸發
if run_button:
    run_backtest()

