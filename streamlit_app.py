import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from math import erf, sqrt
from sklearn.ensemble import HistGradientBoostingRegressor

H, EPS = 24, 1e-8
Z80, Z50 = 1.2816, 0.6745
FEATS = ['v24', 'v72', 'v168', 'ewm72', 'absr1', 'absr6', 'ret24', 'hour']
# Модель проверена (walk-forward на ~13 000 часов) только на BTC и ETH
COINS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
N_PATHS = 12

st.set_page_config(page_title="Крипто-радар", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------
# Данные и прогноз (кэш на 1 час: прогноз обновляется сам)
# ----------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Загружаю данные...")
def get_prices(ticker):
    raw = yf.download(ticker, interval="1h", period="730d", progress=False,
                      auto_adjust=True, timeout=20)
    if raw is None or raw.empty:
        raise RuntimeError("Источник данных не вернул свечи, попробуйте позже.")
    close = raw['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = close.dropna()
    s.index = pd.to_datetime(s.index, utc=True)
    return s[~s.index.duplicated()].sort_index()


def features(price):
    lp = np.log(price)
    r = lp.diff()
    X = pd.DataFrame(index=price.index)
    X['v24'] = r.rolling(24).std()
    X['v72'] = r.rolling(72).std()
    X['v168'] = r.rolling(168).std()
    X['ewm72'] = r.ewm(span=72).std()
    X['absr1'] = r.abs()
    X['absr6'] = r.abs().rolling(6).mean()
    X['ret24'] = lp.diff(24)
    X['hour'] = X.index.hour
    return X, r


@st.cache_data(ttl=3600, show_spinner="Считаю прогноз волатильности...")
def run_forecast(ticker):
    price = get_prices(ticker)
    X, r = features(price)
    T = X.copy()
    T['fut_vol'] = r[::-1].rolling(H).std()[::-1].shift(-1)
    T = T.dropna()
    T['y'] = np.log(T['fut_vol'] + EPS)
    T = T.iloc[:-H]  # отсекаем H строк на границе, чтобы цель не заглядывала вперёд
    model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200)
    model.fit(T[FEATS], T['y'])
    last = X.dropna().iloc[[-1]]
    sigma = float(np.exp(model.predict(last[FEATS]))[0]) * np.sqrt(H)
    return {
        "price": float(price.iloc[-1]),
        "sigma": sigma,
        "hist": price.iloc[-168:],
        "t": price.index[-1],
    }


# ----------------------------------------------------------------------
# Симуляция примеров путей (направление случайное, размах из прогноза)
# ----------------------------------------------------------------------

def simulate_paths(p0, sigma_day, n_paths, seed):
    rng = np.random.default_rng(seed)
    sig_h = sigma_day / np.sqrt(24)
    # t-распределение (4 степени свободы) даёт «толстые хвосты», как у реальной крипты;
    # делим на sqrt(2), чтобы дисперсия осталась равной 1
    shocks = rng.standard_t(4, size=(n_paths, 24)) / np.sqrt(2)
    cum = np.cumsum(shocks * sig_h, axis=1)
    paths = p0 * np.exp(cum)
    return np.hstack([np.full((n_paths, 1), p0), paths])


# ----------------------------------------------------------------------
# График
# ----------------------------------------------------------------------

def make_chart(f, window_h, show_paths, seed):
    s = f["hist"].iloc[-window_h:]
    p0, sig, t0 = f["price"], f["sigma"], s.index[-1]
    hrs = np.arange(0, 25)
    fut = [t0 + pd.Timedelta(hours=int(h)) for h in hrs]
    k = np.sqrt(hrs / 24)
    lo80, hi80 = p0 * np.exp(-Z80 * sig * k), p0 * np.exp(Z80 * sig * k)
    lo50, hi50 = p0 * np.exp(-Z50 * sig * k), p0 * np.exp(Z50 * sig * k)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Цена",
                             line=dict(color="#60a5fa", width=2.5)))
    # диапазон 80%
    fig.add_trace(go.Scatter(x=fut, y=hi80, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut, y=lo80, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(245,158,11,0.22)",
                             name="Диапазон 80%", hoverinfo="skip"))
    # диапазон 50%
    fig.add_trace(go.Scatter(x=fut, y=hi50, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut, y=lo50, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(245,158,11,0.45)",
                             name="Диапазон 50%", hoverinfo="skip"))
    # симуляция возможных путей
    if show_paths:
        paths = simulate_paths(p0, sig, N_PATHS, seed)
        for i in range(N_PATHS):
            fig.add_trace(go.Scatter(
                x=fut, y=paths[i], mode="lines",
                line=dict(color="rgba(226,232,240,0.40)", width=1.2),
                name="Примеры путей (симуляция)", legendgroup="sim",
                showlegend=(i == 0), hoverinfo="skip"))
    # текущая цена
    fig.add_trace(go.Scatter(x=fut, y=[p0] * len(fut), mode="lines",
                             line=dict(color="#e2e8f0", width=1.2, dash="dash"),
                             name="Текущая цена"))
    # подписи границ 80%
    fig.add_trace(go.Scatter(
        x=[fut[-1], fut[-1]], y=[hi80[-1], lo80[-1]], mode="text",
        text=[f"{hi80[-1]:,.0f}", f"{lo80[-1]:,.0f}"],
        textposition=["top left", "bottom left"], showlegend=False,
        textfont=dict(color="#e2e8f0", size=13), hoverinfo="skip"))

    fig.update_layout(
        template="plotly_dark", height=580, hovermode="x unified",
        margin=dict(t=40, l=10, r=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="Цена, USD",
    )
    return fig


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


# ----------------------------------------------------------------------
# Интерфейс
# ----------------------------------------------------------------------

st.title("📊 Крипто-радар")
st.caption("Прогноз размаха движения BTC и ETH на 24 часа. Это не прогноз направления цены "
           "и не финансовый совет.")

if "seed_shift" not in st.session_state:
    st.session_state.seed_shift = 0

with st.sidebar:
    st.header("Настройки")
    coin = st.selectbox("Монета", list(COINS.keys()))
    window_label = st.radio("Окно истории на графике", ["48 часов", "72 часа", "7 дней"], index=1)
    window_h = {"48 часов": 48, "72 часа": 72, "7 дней": 168}[window_label]
    show_paths = st.checkbox("Показать примеры возможных путей (симуляция)", value=True)
    if st.button("🔄 Другие примеры"):
        st.session_state.seed_shift += 1
    st.divider()
    st.subheader("Калькулятор риска")
    balance = st.number_input("Баланс, USD", min_value=10.0, value=1000.0, step=100.0)
    risk_pct = st.slider("Риск на сделку, %", 0.5, 5.0, 2.0, 0.5)
    st.caption("Размер позиции считается так, чтобы при срабатывании стопа вы потеряли "
               "заданный процент баланса.")

try:
    f = run_forecast(COINS[coin])
except Exception as e:
    st.error(f"Не удалось получить данные: {e}")
    st.stop()

price, sig = f["price"], f["sigma"]
age_h = (pd.Timestamp.now(tz="UTC") - f["t"]).total_seconds() / 3600
st.caption(f"Последняя свеча: {f['t']:%d.%m.%Y %H:%M} UTC. Данные и прогноз обновляются раз в час.")
if age_h > 3:
    st.warning(f"Данные устарели на {age_h:.0f} ч: источник задерживает свечи.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Цена", f"{price:,.0f}")
c2.metric("Ожидаемый размах за 24 ч", f"±{sig * 100:.1f}%")
c3.metric("Диапазон 80%",
          f"{price * np.exp(-Z80 * sig):,.0f} – {price * np.exp(Z80 * sig):,.0f}")
c4.metric("Диапазон 50%",
          f"{price * np.exp(-Z50 * sig):,.0f} – {price * np.exp(Z50 * sig):,.0f}")

st.subheader(f"{coin}: цена и диапазон на ближайшие 24 часа")
seed = int(f["t"].timestamp()) + st.session_state.seed_shift
st.plotly_chart(make_chart(f, window_h, show_paths, seed), use_container_width=True)
if show_paths:
    st.caption("Тонкие белые линии: 12 случайных сценариев с размахом из прогноза. Направление "
               "в них случайное: это иллюстрация, как цена может двигаться внутри диапазона, "
               "а не прогноз пути. Кнопка «Другие примеры» в боковой панели рисует новые.")

left, right = st.columns(2)

with left:
    st.subheader("Вероятность остаться в боковике")
    rows = []
    for x in (1, 2, 3, 5):
        p = 2 * phi((x / 100) / sig) - 1
        shown = ">99%" if p > 0.99 else f"{p * 100:.0f}%"
        rows.append({"Диапазон": f"±{x}% от текущей цены",
                     "Вероятность за 24 ч": shown})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Оценка при нормальном распределении. У крипты хвосты толще, поэтому реальная "
               "вероятность резких выходов за рамки выше. Проверено только то, что цена "
               "попадала в 80% диапазон примерно в 80% случаев.")

with right:
    st.subheader("Калькулятор позиции (пример расчёта риска)")
    stop_k = st.select_slider("Стоп-дистанция, в σ за сутки", options=[1.0, 1.5, 2.0], value=1.5)
    stop_pct = stop_k * sig
    size = min(balance * risk_pct / 100 / (price * stop_pct), balance / price)
    st.write(f"Стоп-дистанция: **{stop_pct * 100:.2f}%** от цены.")
    st.write(f"Размер позиции: **{size:.5f} {coin}** "
             f"(риск {risk_pct}% от {balance:,.0f} USD ≈ {balance * risk_pct / 100:,.0f} USD).")
    st.caption("Это расчёт размера под заданный риск, а не сигнал входа или выхода.")

st.divider()
st.caption("Модель прогноза волатильности проверена на BTC и ETH за ~1,5 года (walk-forward) и "
           "превзошла простые базовые методы по ошибке. Направление цены она не предсказывает: "
           "на тех же данных точность направления была близка к случайной. Не финансовый совет, "
           "торговля криптовалютами связана с риском потери средств.")
