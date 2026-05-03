import os
import re
import warnings
import requests
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from groq import Groq
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ─── 종목 리스트 ────────────────────────────────────────────────────────────────

def get_us_tickers() -> list:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(url)[0]
        tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
        print(f"  S&P 500 종목 {len(tickers)}개 로드 완료")
        return tickers
    except Exception:
        print("  Wikipedia 로드 실패 → 기본 리스트 사용")
        return [
            'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','JPM','V',
            'MA','UNH','XOM','LLY','JNJ','PG','HD','COST','MRK','ABBV',
            'CRM','ORCL','AMD','QCOM','TXN','NFLX','DIS','PYPL','ADBE','NOW',
            'PANW','CRWD','ZS','DDOG','SNOW','NET','PLTR','COIN','SMCI','MRVL',
            'MU','AMAT','LRCX','KLAC','ADI','INTC','ON','GS','MS','BAC',
            'WFC','BLK','SPGI','ISRG','REGN','VRTX','GILD','AMGN','PFE','BMY',
            'CAT','DE','HON','GE','RTX','LMT','BA','NEE','XOM','CVX',
            'COP','BKNG','ABNB','UBER','SHOP','SQ','HUBS','WDAY','TWLO','OKTA',
            'WMT','TGT','NKE','SBUX','MCD','PEP','KO','PM','MO','EL',
            'TSLA','GM','F','RIVN','LCID','NIO','PTON','ZM','ROKU','RBLX',
        ]


# ─── 데이터 수집 ────────────────────────────────────────────────────────────────

def fetch_batch_ohlcv(tickers: list) -> dict:
    print(f"  {len(tickers)}개 종목 OHLCV 다운로드 중...")
    try:
        raw = yf.download(
            tickers, period="6mo", interval="1d",
            progress=False, auto_adjust=True, group_by='ticker',
            threads=True
        )
    except Exception as e:
        print(f"  배치 다운로드 실패: {e}")
        return {}

    result = {}
    for ticker in tickers:
        try:
            df = raw[ticker].copy() if len(tickers) > 1 else raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            if len(df) >= 60:
                result[ticker] = df
        except Exception:
            pass

    print(f"  유효 종목: {len(result)}개")
    return result


def fetch_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'forward_pe': info.get('forwardPE', None),
            'trailing_pe': info.get('trailingPE', None),
            'revenue_growth': info.get('revenueGrowth', None),
            'earnings_growth': info.get('earningsGrowth', None),
            'profit_margin': info.get('profitMargins', None),
            'roe': info.get('returnOnEquity', None),
            '52w_high': info.get('fiftyTwoWeekHigh', None),
            '52w_low': info.get('fiftyTwoWeekLow', None),
            'analyst_target': info.get('targetMeanPrice', None),
            'recommendation': info.get('recommendationKey', 'N/A'),
        }
    except Exception:
        return {'name': ticker, 'sector': 'N/A'}


# ─── 기술적 분석 ────────────────────────────────────────────────────────────────

def calc_tech_score(df: pd.DataFrame) -> dict:
    score = 0
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    price = float(c.iloc[-1])

    # EMA 계산
    ema9  = float(ta.trend.ema_indicator(c, window=9).iloc[-1])
    ema21 = float(ta.trend.ema_indicator(c, window=21).iloc[-1])
    ema50 = float(ta.trend.ema_indicator(c, window=50).iloc[-1])
    ema200 = float(ta.trend.ema_indicator(c, window=200).iloc[-1]) if len(df) >= 200 else None

    # EMA 정배열 (25점)
    if price > ema50:  score += 5
    if price > ema21:  score += 5
    if ema9 > ema21:   score += 5
    if ema21 > ema50:  score += 5
    if ema200 and price > ema200: score += 5

    # MACD (20점)
    macd_obj  = ta.trend.MACD(c, window_fast=12, window_slow=26, window_sign=9)
    macd      = float(macd_obj.macd().iloc[-1])
    sig       = float(macd_obj.macd_signal().iloc[-1])
    hist      = float(macd_obj.macd_diff().iloc[-1])
    prev_hist = float(macd_obj.macd_diff().iloc[-2])
    if macd > sig:        score += 10
    if hist > prev_hist:  score += 10

    # RSI (20점)
    rsi = float(ta.momentum.RSIIndicator(c, window=14).rsi().iloc[-1])
    if 40 <= rsi <= 65:        score += 20
    elif 30 <= rsi < 40 or 65 < rsi <= 72: score += 10

    # 거래량 급증 (20점)
    avg_vol = float(v.rolling(20).mean().iloc[-1])
    cur_vol = float(v.iloc[-1])
    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio >= 1.5:   score += 20
    elif vol_ratio >= 1.2: score += 10

    # ADX 추세 강도 (15점)
    adx = float(ta.trend.ADXIndicator(h, l, c, window=14).adx().iloc[-1])
    if adx >= 25:   score += 15
    elif adx >= 20: score += 8
    elif adx >= 15: score += 4

    # 1개월 수익률 모멘텀 보너스
    ret_1m = (price / float(c.iloc[-21]) - 1) * 100 if len(c) >= 21 else 0
    ret_3m = (price / float(c.iloc[-63]) - 1) * 100 if len(c) >= 63 else 0
    if ret_1m > 8:  score += 5
    elif ret_1m > 3: score += 2

    # BB 위치 (하단 근처 = 반등 기대)
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    bb_upper = float(bb.bollinger_hband().iloc[-1])
    bb_lower = float(bb.bollinger_lband().iloc[-1])
    bb_mid   = float(bb.bollinger_mavg().iloc[-1])
    bb_pct = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

    # 52주 신고가 근처 제외 (과열 방지)
    recent_high = float(c.tail(252).max()) if len(c) >= 252 else float(c.max())
    if price >= recent_high * 0.99:
        score -= 10

    return {
        'score': score,
        'price': price,
        'ema9': ema9, 'ema21': ema21, 'ema50': ema50, 'ema200': ema200,
        'rsi': rsi,
        'macd': macd, 'macd_signal': sig, 'macd_hist': hist,
        'adx': adx,
        'vol_ratio': vol_ratio,
        'ret_1m': ret_1m,
        'ret_3m': ret_3m,
        'bb_pct': bb_pct,
        'bb_upper': bb_upper, 'bb_lower': bb_lower,
    }


# ─── 리포트 생성 ────────────────────────────────────────────────────────────────

def remove_cjk(text: str) -> str:
    return re.sub(r'[一-鿿぀-ヿ㐀-䶿豈-﫿]', '', text)


def build_stock_summary(ranked: list) -> str:
    lines = [f"=== 미국 주식 추천 종목 데이터 [{datetime.now().strftime('%Y-%m-%d')}] ===\n"]
    for i, (ticker, tech, fund) in enumerate(ranked, 1):
        pe = fund.get('forward_pe') or fund.get('trailing_pe')
        rev_g = fund.get('revenue_growth')
        eps_g = fund.get('earnings_growth')
        target = fund.get('analyst_target')
        upside = ((target / tech['price'] - 1) * 100) if target and tech['price'] else None

        lines.append(f"""
[{i}위] {ticker} ({fund.get('name', ticker)})
섹터: {fund.get('sector', 'N/A')}
현재가: ${tech['price']:,.2f}
기술점수: {tech['score']}점

기술적 지표:
  EMA 9/21/50/200: ${tech['ema9']:,.2f} / ${tech['ema21']:,.2f} / ${tech['ema50']:,.2f} / {'$'+f"{tech['ema200']:,.2f}" if tech['ema200'] else 'N/A'}
  EMA 정렬: {'정배열(상승)' if tech['ema9'] > tech['ema21'] > tech['ema50'] else '혼조'}
  RSI(14): {tech['rsi']:.1f}
  MACD: {tech['macd']:.3f} | Signal: {tech['macd_signal']:.3f} | Hist: {'양수(강세)' if tech['macd_hist'] > 0 else '음수(약세)'}
  ADX: {tech['adx']:.1f} | {'강한 추세' if tech['adx'] >= 25 else '추세 형성 중'}
  거래량 비율: {tech['vol_ratio']:.2f}x (20일 평균 대비)
  BB 위치: {tech['bb_pct']*100:.0f}% (0%=하단, 100%=상단)
  1개월 수익률: {tech['ret_1m']:+.1f}%
  3개월 수익률: {tech['ret_3m']:+.1f}%

펀더멘탈:
  PER: {f'{pe:.1f}' if pe else 'N/A'}
  매출 성장률: {f'{rev_g*100:.1f}%' if rev_g else 'N/A'}
  EPS 성장률: {f'{eps_g*100:.1f}%' if eps_g else 'N/A'}
  시가총액: {'$'+f'{fund.get("market_cap",0)/1e9:.0f}B' if fund.get('market_cap') else 'N/A'}
  애널리스트 목표가: {'$'+f'{target:,.2f}' if target else 'N/A'}{f' (상승여력 {upside:.1f}%)' if upside else ''}
  투자의견: {fund.get('recommendation', 'N/A').upper()}
""")
    return '\n'.join(lines)


def generate_report(ranked: list) -> str:
    summary = build_stock_summary(ranked)
    today = datetime.now().strftime('%Y년 %m월 %d일')
    weekday = ['월', '화', '수', '목', '금', '토', '일'][datetime.now().weekday()]

    prompt = f"""당신은 월스트리트 출신 10년 경력의 미국 주식 전문 애널리스트입니다.
한국 개인 투자자(토스증권 사용)를 위해 아래 종목 데이터를 분석하여 투자 리포트를 작성해주세요.
반드시 순수한 한국어만 사용하세요. 한자, 일본어, 중국어는 절대 사용하지 마세요.

{summary}

아래 형식으로 작성해주세요:

📊 미국 주식 일일 투자 추천 리포트
{today} ({weekday}요일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

오늘의 시장 한줄 요약: [현재 미국 시장 상황 1~2문장]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 오늘의 추천 종목 TOP 10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(각 종목별로 아래 형식 사용)

[순위]️ 티커 (회사명) | 섹터
현재가: $XX.XX
💰 추천 매수단가: $XX.XX ~ $XX.XX
🎯 목표가: $XX.XX (+XX%) | 기간: X~X개월
🛡️ 손절가: $XX.XX (-XX%)

📌 투자 근거 (3가지):
• [펀더멘탈 근거]
• [성장 동력 / 촉매 요인]
• [섹터/매크로 관점]

📈 차트 분석:
• 추세: [EMA/MACD 기반 분석]
• 모멘텀: [RSI/거래량 분석]
• 진입 시점: [지금 들어가야 하는 이유]

─────────────────────────────

(10개 모두 작성)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 투자 유의사항
• 본 리포트는 참고용이며 투자 손익은 본인 책임입니다.
• 토스증권에서 거래 시 환율 변동 리스크를 고려하세요.
• 분산 투자를 권장합니다.
"""

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "당신은 한국어로만 응답하는 미국 주식 전문 애널리스트입니다. 반드시 순수한 한국어(한글)와 영어/숫자만 사용하세요. 한자, 일본어, 중국어는 절대 사용하지 마세요."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=6000,
    )
    return remove_cjk(response.choices[0].message.content)


# ─── 전송 ────────────────────────────────────────────────────────────────────────

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)]
    for i, chunk in enumerate(chunks):
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=10)
        print(f"  텔레그램 전송 [{i+1}/{len(chunks)}]: {'✅' if r.ok else '❌ ' + str(r.json())}")


# ─── 메인 ────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*55}")
    print(f"미국 주식 추천 리포트 [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"{'='*55}\n")

    print("[1/5] 종목 리스트 수집 중...")
    tickers = get_us_tickers()

    print("[2/5] OHLCV 데이터 다운로드 중...")
    ohlcv_data = fetch_batch_ohlcv(tickers)
    if not ohlcv_data:
        print("데이터 수집 실패")
        return

    print("[3/5] 기술적 지표 계산 및 종목 선별 중...")
    scores = {}
    for ticker, df in ohlcv_data.items():
        try:
            scores[ticker] = calc_tech_score(df)
        except Exception:
            pass

    # 기술점수 상위 30개 선별
    top30 = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)[:30]
    print(f"  기술점수 상위 30개 선별 완료")

    print("[4/5] 펀더멘탈 데이터 수집 중 (상위 30개)...")
    ranked = []
    for ticker, tech in top30:
        fund = fetch_fundamentals(ticker)
        # 시가총액 10억 달러 미만 제외
        if fund.get('market_cap', 0) > 1e9:
            ranked.append((ticker, tech, fund))

    # 최종 10개: 기술점수 + 애널리스트 목표가 상승여력 기반 정렬
    def final_score(item):
        ticker, tech, fund = item
        s = tech['score']
        target = fund.get('analyst_target')
        if target and tech['price'] > 0:
            upside = (target / tech['price'] - 1) * 100
            s += min(upside * 0.3, 15)  # 상승여력 보너스 최대 15점
        return s

    ranked = sorted(ranked, key=final_score, reverse=True)[:10]
    print(f"  최종 추천 10개 종목: {[t for t, _, _ in ranked]}")

    print("[5/5] AI 분석 리포트 생성 중...")
    report = generate_report(ranked)

    print("\n📨 텔레그램 전송 중...")
    send_telegram(report)

    print("\n✅ 완료!\n")


if __name__ == "__main__":
    main()
