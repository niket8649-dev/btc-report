import os
import warnings
import requests
import pandas as pd
import ta
import yfinance as yf
from groq import Groq
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
REPORTS_DIR = Path.home() / "Documents" / "btc_reports"
warnings.filterwarnings("ignore")


# ─── 데이터 수집 ────────────────────────────────────────────────────────────────

def fetch_ohlcv(timeframe: str, limit: int = 300) -> pd.DataFrame:
    if timeframe == "1d":
        df = yf.download("BTC-USD", period="400d", interval="1d", progress=False, auto_adjust=True)
    elif timeframe == "4h":
        raw = yf.download("BTC-USD", period="60d", interval="1h", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        df = raw.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    else:
        df = yf.download("BTC-USD", period="7d", interval="1h", progress=False, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna().tail(limit)
    return df


def fetch_futures_data() -> dict:
    result = {}

    # CoinGecko - 현재가 및 시장 데이터 (미국 IP 차단 없음)
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin",
            params={"localization": "false", "tickers": "false", "community_data": "false"},
            timeout=15
        ).json()
        md = r.get("market_data", {})
        result["price"] = md.get("current_price", {}).get("usd", 0)
        result["price_change_24h"] = md.get("price_change_percentage_24h", 0)
        result["volume_24h"] = md.get("total_volume", {}).get("usd", 0)
    except Exception:
        pass

    # Coinglass - 펀딩비 / OI / 롱숏 (공개 API)
    try:
        headers = {"accept": "application/json"}
        r = requests.get("https://open-api.coinglass.com/public/v2/funding", params={"symbol": "BTC"}, headers=headers, timeout=10).json()
        if r.get("data"):
            rates = [float(x.get("rate", 0)) for x in r["data"] if x.get("rate")]
            if rates:
                result["funding_rate"] = rates[0]
                result["avg_funding_8h"] = sum(rates[:10]) / min(len(rates), 10)
    except Exception:
        pass

    try:
        r = requests.get("https://open-api.coinglass.com/public/v2/open_interest", params={"symbol": "BTC"}, timeout=10).json()
        if r.get("data"):
            total_oi = sum(float(x.get("openInterest", 0)) for x in r["data"])
            result["oi_value"] = total_oi
    except Exception:
        pass

    try:
        r = requests.get("https://open-api.coinglass.com/public/v2/long_short", params={"symbol": "BTC", "interval": "1h"}, timeout=10).json()
        if r.get("data") and r["data"]:
            latest = r["data"][0]
            result["long_ratio"] = float(latest.get("longRatio", 0)) * 100
            result["short_ratio"] = float(latest.get("shortRatio", 0)) * 100
    except Exception:
        pass

    return result


# ─── 지표 계산 ───────────────────────────────────────────────────────────────────

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # EMA
    for p in [9, 21, 50, 200]:
        df[f"ema{p}"] = ta.trend.ema_indicator(c, window=p)

    # MACD
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # RSI & StochRSI
    df["rsi"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    srsi = ta.momentum.StochRSIIndicator(c, window=14)
    df["srsi_k"] = srsi.stochrsi_k()
    df["srsi_d"] = srsi.stochrsi_d()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_u"] = bb.bollinger_hband()
    df["bb_m"] = bb.bollinger_mavg()
    df["bb_l"] = bb.bollinger_lband()
    df["bb_w"] = bb.bollinger_wband()

    # ATR & ADX
    df["atr"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"] = adx.adx()
    df["di_p"] = adx.adx_pos()
    df["di_m"] = adx.adx_neg()

    # Ichimoku
    ichi = ta.trend.IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
    df["tenkan"] = ichi.ichimoku_conversion_line()
    df["kijun"] = ichi.ichimoku_base_line()
    df["span_a"] = ichi.ichimoku_a()
    df["span_b"] = ichi.ichimoku_b()

    # OBV, CCI, Williams %R, MFI
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["cci"] = ta.trend.CCIIndicator(h, l, c, window=20).cci()
    df["willr"] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()
    df["mfi"] = ta.volume.MFIIndicator(h, l, c, v, window=14).money_flow_index()

    return df


def find_sr_levels(df: pd.DataFrame, window: int = 10) -> tuple:
    price = df["close"].iloc[-1]
    recent = df.tail(120)
    supports, resistances = [], []

    for i in range(window, len(recent) - window):
        h = recent["high"].iloc[i]
        lo = recent["low"].iloc[i]
        if h == recent["high"].iloc[i - window:i + window + 1].max():
            resistances.append(round(h, 0))
        if lo == recent["low"].iloc[i - window:i + window + 1].min():
            supports.append(round(lo, 0))

    sup = sorted(set(s for s in supports if s < price), reverse=True)[:4]
    res = sorted(set(r for r in resistances if r > price))[:4]
    return sup, res


def calc_fibonacci(df: pd.DataFrame) -> dict:
    recent = df.tail(90)
    hi, lo = recent["high"].max(), recent["low"].min()
    diff = hi - lo
    return {
        "0.0%":  round(hi, 0),
        "23.6%": round(hi - 0.236 * diff, 0),
        "38.2%": round(hi - 0.382 * diff, 0),
        "50.0%": round(hi - 0.500 * diff, 0),
        "61.8%": round(hi - 0.618 * diff, 0),
        "78.6%": round(hi - 0.786 * diff, 0),
        "100%":  round(lo, 0),
    }


# ─── Claude 리포트 생성 ──────────────────────────────────────────────────────────

def build_data_summary(df_1d, df_4h, df_1h, fd) -> str:
    d = df_1d.iloc[-1]
    h4 = df_4h.iloc[-1]
    h1 = df_1h.iloc[-1]

    sup, res = find_sr_levels(df_1d)
    fib = calc_fibonacci(df_1d)

    ema_align = (
        "완전 상승 정렬" if d.ema9 > d.ema21 > d.ema50 > d.ema200
        else "완전 하락 정렬" if d.ema9 < d.ema21 < d.ema50 < d.ema200
        else "혼조"
    )
    obv_trend = "상승" if df_1d["obv"].iloc[-1] > df_1d["obv"].iloc[-5] else "하락"
    price_vs_cloud = (
        "구름 위 (강세)" if d.close > max(d.span_a or 0, d.span_b or 0)
        else "구름 아래 (약세)" if d.close < min(d.span_a or 0, d.span_b or 0)
        else "구름 안 (중립)"
    ) if d.get("span_a") and d.get("span_b") else "N/A"

    def v(x, fmt=".2f"):
        try:
            return format(float(x), fmt) if x is not None and not pd.isna(float(x)) else "N/A"
        except Exception:
            return "N/A"

    price = fd.get("price", d.close)

    return f"""
=== BTC/USDT 시장 데이터 [{datetime.now().strftime('%Y-%m-%d %H:%M')} KST] ===

[현재 시장]
현재가: ${price:,.2f}
24H 변동: {v(fd.get('price_change_24h'))}%
24H 거래량: {v(fd.get('volume_24h'), ',.0f')} USDT

[일봉 기술적 지표]
EMA 9/21/50/200: ${v(d.ema9,',.0f')} / ${v(d.ema21,',.0f')} / ${v(d.ema50,',.0f')} / ${v(d.ema200,',.0f')}
EMA 정렬: {ema_align}
MACD: {v(d.macd)} | Signal: {v(d.macd_sig)} | Hist: {v(d.macd_hist)}
MACD 상태: {'골든크로스' if (d.macd or 0) > (d.macd_sig or 0) else '데드크로스'}
RSI(14): {v(d.rsi)}
StochRSI K/D: {v(d.srsi_k)} / {v(d.srsi_d)}
BB 상/중/하: ${v(d.bb_u,',.0f')} / ${v(d.bb_m,',.0f')} / ${v(d.bb_l,',.0f')} | Width: {v(d.bb_w)}%
ATR(14): ${v(d.atr,',.0f')}
ADX: {v(d.adx)} | DI+: {v(d.di_p)} | DI-: {v(d.di_m)}
CCI(20): {v(d.cci)} | Williams%R: {v(d.willr)} | MFI: {v(d.mfi)}
Ichimoku Tenkan/Kijun: ${v(d.tenkan,',.0f')} / ${v(d.kijun,',.0f')}
가격 vs 구름: {price_vs_cloud}
OBV 추세: {obv_trend}

[4시간봉 핵심]
RSI: {v(h4.rsi)} | MACD: {v(h4.macd)} | ADX: {v(h4.adx)}
EMA 21/50: ${v(h4.ema21,',.0f')} / ${v(h4.ema50,',.0f')}

[1시간봉 핵심]
RSI: {v(h1.rsi)} | MACD: {v(h1.macd)}
EMA 21/50: ${v(h1.ema21,',.0f')} / ${v(h1.ema50,',.0f')}

[지지/저항]
저항: {' | '.join(f'${int(r):,}' for r in res) or 'N/A'}
지지: {' | '.join(f'${int(s):,}' for s in sup) or 'N/A'}

[피보나치 (90일)]
{chr(10).join(f'{k}: ${int(val):,}' for k, val in fib.items())}

[선물 시장]
펀딩비 현재: {v(fd.get('funding_rate'), '.4f')}%
펀딩비 평균(10회): {v(fd.get('avg_funding_8h'), '.4f')}%
미결제약정: ${fd.get('oi_value', 0)/1e9:.2f}B
OI 24H 변화: {v(fd.get('oi_change_24h'))}%
롱/숏 비율: {v(fd.get('long_ratio'), '.1f')}% / {v(fd.get('short_ratio'), '.1f')}%
"""


def generate_report(df_1d, df_4h, df_1h, fd) -> str:
    summary = build_data_summary(df_1d, df_4h, df_1h, fd)

    prompt = f"""당신은 10년 경력의 비트코인 선물 전문 트레이더 겸 퀀트 분석가입니다.
아래 데이터를 기반으로 전문 트레이딩 리포트를 작성해주세요. 사용자는 Bybit 선물 트레이더입니다.

{summary}

아래 형식으로 작성해주세요:

📊 BTC/USDT 일일 선물 트레이딩 리포트
{datetime.now().strftime('%Y년 %m월 %d일')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ 시장 종합 판단
• 단기(1H/4H):
• 중기(1D):
• 시장 심리:

2️⃣ 상승/하락 확률
📈 상승 확률: XX%
📉 하락 확률: XX%
▶ 핵심 근거:
  - (지표 근거 3~5가지)

3️⃣ 핵심 지표 해석
• 모멘텀(RSI/StochRSI):
• MACD:
• 볼린저밴드:
• Ichimoku:
• ADX/추세 강도:
• 펀딩비 분석:
• OI 분석:
• 롱숏 비율:

4️⃣ 트레이딩 전략

🟢 롱(매수) 전략
  진입가: $XX,XXX ~ $XX,XXX
  목표가 1: $XX,XXX (+X.X%)
  목표가 2: $XX,XXX (+X.X%)
  손절가: $XX,XXX (-X.X%)
  추천 레버리지: Xx
  진입 조건: [구체적 조건]

🔴 숏(매도) 전략
  진입가: $XX,XXX ~ $XX,XXX
  목표가 1: $XX,XXX (-X.X%)
  목표가 2: $XX,XXX (-X.X%)
  손절가: $XX,XXX (+X.X%)
  추천 레버리지: Xx
  진입 조건: [구체적 조건]

5️⃣ 리스크 요인
• [리스크 1]
• [리스크 2]
• [리스크 3]

6️⃣ 핵심 레벨
  절대 지지: $XX,XXX
  절대 저항: $XX,XXX
  핵심 변곡점: $XX,XXX

7️⃣ 오늘의 핵심 메시지
[2~3문장]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 본 리포트는 참고용이며, 투자 결정은 본인 책임입니다.
"""

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "당신은 한국어로만 응답하는 비트코인 선물 트레이딩 전문가입니다. 반드시 순수한 한국어만 사용하세요. 한자, 일본어, 중국어는 절대 사용하지 마세요. 모든 텍스트는 한국어(한글)와 영어 숫자로만 작성하세요."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ─── 전송 & 저장 ─────────────────────────────────────────────────────────────────

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)]
    for i, chunk in enumerate(chunks):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk}
        r = requests.post(url, json=payload, timeout=10)
        print(f"  텔레그램 전송 [{i+1}/{len(chunks)}]: {'✅' if r.ok else '❌ ' + str(r.json())}")


def save_report(text: str) -> Path:
    # GitHub Actions 환경에서는 로컬 저장 건너뜀
    if os.getenv("GITHUB_ACTIONS"):
        print("  파일 저장: GitHub Actions 환경 - 건너뜀")
        return Path(".")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"BTC_{datetime.now().strftime('%Y-%m-%d')}.md"
    path.write_text(text, encoding="utf-8")
    print(f"  파일 저장: {path}")
    return path


# ─── 메인 ────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*50}")
    print(f"BTC 트레이딩 리포트 생성 [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"{'='*50}\n")

    print("[1/4] 시장 OHLCV 데이터 수집 중...")
    df_1d = fetch_ohlcv("1d", 300)
    df_4h = fetch_ohlcv("4h", 200)
    df_1h = fetch_ohlcv("1h", 100)

    print("[2/4] 선물 시장 데이터 수집 중...")
    fd = fetch_bybit_futures_data()

    print("[3/4] 기술적 지표 계산 중...")
    df_1d = calc_indicators(df_1d)
    df_4h = calc_indicators(df_4h)
    df_1h = calc_indicators(df_1h)

    print("[4/4] Claude AI 분석 리포트 생성 중...")
    report = generate_report(df_1d, df_4h, df_1h, fd)

    print("\n📁 파일 저장 중...")
    save_report(report)

    print("📨 텔레그램 전송 중...")
    send_telegram(report)

    print("\n✅ 완료!\n")


def fetch_bybit_futures_data():
    return fetch_futures_data()


if __name__ == "__main__":
    main()
