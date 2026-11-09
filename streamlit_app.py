import os
import sys
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------------------------------------------------------------
# Настройки
# ----------------------------------------------------------------------
H, EPS = 24, 1e-8                  # горизонт прогноза, часов
Z80, Z50 = 1.2816, 0.6745          # границы интервалов 80% и 50% (нормальное приближение)
FEATS = ['v24', 'v72', 'v168', 'ewm72', 'absr1', 'absr6', 'ret24', 'hour']
LOG = "forecast_log.csv"
LOG_COLS = ['time_utc', 'coin', 'price', 'sigma_24h', 'low80', 'high80']
COINS = {"BTC-USD": "BTC", "ETH-USD": "ETH"}
MIN_CHECKS = 20                    # трекер показываем как процент только от этого числа проверок
STALE_H = 3                        # данные старше N часов считаем устаревшими


# ----------------------------------------------------------------------
# Данные
# ----------------------------------------------------------------------
def get_prices(ticker, attempts=3):
    last_err = None
    for i in range(attempts):
        try:
            raw = yf.download(ticker, interval="1h", period="730d",
                              progress=False, auto_adjust=True, timeout=20)
            if raw is None or raw.empty:
                raise RuntimeError("пустой ответ")
            c = raw["Close"]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            c = c.dropna()
            c.index = pd.to_datetime(c.index, utc=True)
            return c[~c.index.duplicated()].sort_index()
        except Exception as e:
            last_err = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"{ticker}: не удалось получить данные: {last_err}")


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


# ----------------------------------------------------------------------
# Модель (логика проверена walk-forward на ~13 000 часов, не менять без нового теста)
# ----------------------------------------------------------------------
def forecast(price):
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
    p = float(price.iloc[-1])
    return {
        'time_utc': str(price.index[-1]),
        'price': p,
        'sigma_24h': sigma,
        'low80': p * np.exp(-Z80 * sigma),
        'high80': p * np.exp(Z80 * sigma),
    }


# ----------------------------------------------------------------------
# Лог и трекер
# ----------------------------------------------------------------------
def load_log():
    if os.path.exists(LOG):
        log = pd.read_csv(LOG)
        for col in LOG_COLS:
            if col not in log.columns:
                log[col] = np.nan
        return log[LOG_COLS]
    return pd.DataFrame(columns=LOG_COLS)


def evaluate(log, coin, price):
    """Проверка прошлых прогнозов: попала ли цена через 24 часа в диапазон 80%.
    В статистику идёт не больше одного прогноза в сутки на монету (первый за день),
    чтобы ручные перезапуски не раздували число проверок."""
    g = log[log['coin'] == coin].copy()
    g['ts'] = pd.to_datetime(g['time_utc'], utc=True, errors='coerce')
    g = g.dropna(subset=['ts', 'low80', 'high80']).sort_values('ts')
    g['day'] = g['ts'].dt.date
    g = g.drop_duplicates('day', keep='first')
    hits = n = 0
    last = None
    for _, r in g.iterrows():
        t = r['ts'] + pd.Timedelta(hours=H)
        if t > price.index[-1]:
            continue
        actual = float(price.loc[:t].iloc[-1])
        hit = bool(float(r['low80']) <= actual <= float(r['high80']))
        n += 1
        hits += int(hit)
        if last is None or t > last['t']:
            last = {'t': t, 'low': float(r['low80']), 'high': float(r['high80']),
                    'actual': actual, 'hit': hit}
    return hits, n, last


# ----------------------------------------------------------------------
# График
# ----------------------------------------------------------------------
def make_chart(prices, forecasts, path="chart.png"):
    bg, fg = "#0f172a", "#e2e8f0"
    fig, axes = plt.subplots(2, 1, figsize=(8, 9.5), dpi=150)
    fig.patch.set_facecolor(bg)
    for ax, (coin, short) in zip(axes, COINS.items()):
        s = prices[coin].iloc[-72:]
        f = forecasts[coin]
        p0, sig, t0 = f['price'], f['sigma_24h'], s.index[-1]
        hrs = np.arange(0, H + 1)
        fut = [t0 + pd.Timedelta(hours=int(h)) for h in hrs]
        k = np.sqrt(hrs / H)
        lo80, hi80 = p0 * np.exp(-Z80 * sig * k), p0 * np.exp(Z80 * sig * k)
        lo50, hi50 = p0 * np.exp(-Z50 * sig * k), p0 * np.exp(Z50 * sig * k)

        ax.set_facecolor(bg)
        ax.plot(s.index, s.values, color="#60a5fa", lw=2.2, label="Цена, 72 ч")
        ax.fill_between(fut, lo80, hi80, color="#f59e0b", alpha=0.25, label="Диапазон 80%")
        ax.fill_between(fut, lo50, hi50, color="#f59e0b", alpha=0.45, label="Диапазон 50%")
        ax.plot(fut, [p0] * len(fut), color=fg, lw=1, ls="--")
        ax.margins(y=0.12)
        ax.annotate(f"{hi80[-1]:,.0f}", (fut[-1], hi80[-1]), color=fg, fontsize=12,
                    xytext=(-6, 8), textcoords="offset points", ha="right")
        ax.annotate(f"{lo80[-1]:,.0f}", (fut[-1], lo80[-1]), color=fg, fontsize=12,
                    xytext=(-6, -18), textcoords="offset points", ha="right")
        ax.set_title(f"{short}: ±{sig * 100:.1f}% за 24 ч", color=fg, fontsize=16, pad=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        ax.tick_params(colors=fg, labelsize=10)
        ax.tick_params(axis="x", rotation=20)
        for sp in ax.spines.values():
            sp.set_color("#334155")
        ax.grid(color="#1e293b", lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, facecolor=bg,
               edgecolor="#334155", labelcolor=fg, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


# ----------------------------------------------------------------------
# Telegram
# ----------------------------------------------------------------------
def send_telegram(text, img_path):
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if not (token and chat):
        print("Telegram: секреты не заданы, отправка пропущена")
        return True
    api = f"https://api.telegram.org/bot{token}"
    for attempt in range(3):
        try:
            with open(img_path, "rb") as img:
                if len(text) <= 1000:   # лимит подписи к фото в Telegram: 1024 символа
                    resp = requests.post(f"{api}/sendPhoto",
                                         data={"chat_id": chat, "caption": text},
                                         files={"photo": img}, timeout=30)
                else:
                    resp = requests.post(f"{api}/sendPhoto", data={"chat_id": chat},
                                         files={"photo": img}, timeout=30)
                    if resp.ok:
                        resp = requests.post(f"{api}/sendMessage",
                                             data={"chat_id": chat, "text": text}, timeout=20)
            print("Telegram:", resp.status_code)
            if resp.ok:
                return True
            print(resp.text[:300])
        except Exception as e:
            # печатаем только тип ошибки: текст исключения может содержать URL с токеном
            print("Telegram: ошибка запроса:", type(e).__name__)
        time.sleep(5 * (attempt + 1))
    return False


# ----------------------------------------------------------------------
# Основной сценарий
# ----------------------------------------------------------------------
def main():
    log = load_log()
    prices = {c: get_prices(c) for c in COINS}

    forecasts, rows, blocks = {}, [], []
    tot_hits = tot_n = 0
    max_age = 0.0
    now = pd.Timestamp.now(tz="UTC")

    for c, short in COINS.items():
        f = forecast(prices[c])
        forecasts[c] = f
        rows.append({'coin': c, **f})
        max_age = max(max_age, (now - prices[c].index[-1]).total_seconds() / 3600)

        hits, n, last = evaluate(log, c, prices[c])
        tot_hits += hits
        tot_n += n

        lines = [f"{short}  {f['price']:,.0f}",
                 f"Диапазон 24ч (80%): {f['low80']:,.0f} – {f['high80']:,.0f}  "
                 f"(±{f['sigma_24h'] * 100:.1f}%)"]
        if last:
            mark = "✅ попал" if last['hit'] else "❌ мимо"
            lines.append(f"Прошлый прогноз: {mark} (факт {last['actual']:,.0f}, "
                         f"диапазон {last['low']:,.0f} – {last['high']:,.0f})")
        blocks.append("\n".join(lines))

    dt = pd.to_datetime(next(iter(forecasts.values()))['time_utc'])
    head = f"📊 Радар волатильности | {dt:%d.%m %H:%M} UTC\n\n"
    if tot_n >= MIN_CHECKS:
        trk = f"Попаданий в 80% диапазон: {tot_hits} из {tot_n} ({tot_hits / tot_n * 100:.0f}%)"
    else:
        trk = f"Трекер попаданий: копим статистику ({tot_n} из {MIN_CHECKS} проверок)"
    warn = (f"⚠️ Данные биржи задерживаются на {max_age:.0f} ч\n"
            if max_age > STALE_H else "")
    text = (head + "\n\n".join(blocks) + "\n\n" + warn + trk +
            "\nПрогноз размаха движения, не направления. Не финансовый совет.")
    print(text)

    # лог: добавляем новые строки, дубликаты (монета + время свечи) заменяем
    new = pd.concat([log, pd.DataFrame(rows)], ignore_index=True)
    new = new.drop_duplicates(subset=['coin', 'time_utc'], keep='last')
    new[LOG_COLS].to_csv(LOG, index=False)

    make_chart(prices, forecasts)
    ok = send_telegram(text, "chart.png")
    if not ok:
        print("Не удалось отправить пост в Telegram")
        sys.exit(1)


if __name__ == "__main__":
    main()
