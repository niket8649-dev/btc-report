import os
import json
import warnings
import requests
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

TOTAL_CAPITAL  = 2378    # 보유 USDT
RISK_PCT       = 0.02    # 트레이드당 리스크 2%
MAX_POSITION_PCT = 0.30  # 최대 포지션 30%
DEFAULT_LEVERAGE = 5
COOLDOWN_HOURS = 4       # 같은 방향 재알림 방지

# 로컬: ~/.btc_alert_state.json / GitHub Actions: repo 내 파일
IS_GHA     = bool(os.getenv("GITHUB_ACTIONS"))
STATE_FILE = Path("btc_alert_state.json") if IS_GHA else Path.home() / ".btc_alert_state.json"


# ─── 상태 관리 ────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, default=str, indent=2))


# ─── 데이터 수집 ────────────────────────────────────────────────────────────────

def fetch_data() -> tuple:
    raw_1h = yf.download("BTC-USD", period="7d",  interval="1h", progress=False, auto_adjust=True)
    raw_4h = yf.download("BTC-USD", period="30d", interval="1h", progress=False, auto_adjust=True)

    for df in [raw_1h, raw_4h]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [c.lower() for c in df.columns]

    df_4h = raw_4h.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    return raw_1h.dropna(), df_4h


# ─── 신호 감지 ────────────────────────────────────────────────────────────────

def check_signals(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> tuple:
    c1 = df_1h["close"]
    c4 = df_4h["close"]
    price = float(c1.iloc[-1])

    # ── 1H 지표 ──
    rsi_1h   = float(ta.momentum.RSIIndicator(c1, window=14).rsi().iloc[-1])
    macd_obj = ta.trend.MACD(c1, window_fast=12, window_slow=26, window_sign=9)
    macd_1h  = float(macd_obj.macd().iloc[-1])
    sig_1h   = float(macd_obj.macd_signal().iloc[-1])
    macd_prev = float(macd_obj.macd().iloc[-2])
    sig_prev  = float(macd_obj.macd_signal().iloc[-2])
    bb       = ta.volatility.BollingerBands(c1, window=20)
    bb_upper = float(bb.bollinger_hband().iloc[-1])
    bb_lower = float(bb.bollinger_lband().iloc[-1])
    ema21_1h = float(ta.trend.ema_indicator(c1, window=21).iloc[-1])
    ema50_1h = float(ta.trend.ema_indicator(c1, window=50).iloc[-1])

    # ── 4H 지표 ──
    rsi_4h    = float(ta.momentum.RSIIndicator(c4, window=14).rsi().iloc[-1])
    macd4_obj = ta.trend.MACD(c4)
    macd_4h   = float(macd4_obj.macd().iloc[-1])
    sig_4h    = float(macd4_obj.macd_signal().iloc[-1])
    adx_4h    = float(ta.trend.ADXIndicator(df_4h["high"], df_4h["low"], c4, window=14).adx().iloc[-1])

    indicators = {
        "price": price,
        "rsi_1h": rsi_1h, "rsi_4h": rsi_4h,
        "macd_1h": macd_1h, "sig_1h": sig_1h,
        "macd_4h": macd_4h, "sig_4h": sig_4h,
        "bb_upper": bb_upper, "bb_lower": bb_lower,
        "ema21_1h": ema21_1h, "ema50_1h": ema50_1h,
        "adx_4h": adx_4h,
    }

    long_conds, short_conds = [], []

    # ── 롱 조건 ──
    if rsi_1h < 35:
        long_conds.append(f"RSI(1H) 과매도 {rsi_1h:.1f} (<35)")
    if macd_prev < sig_prev and macd_1h > sig_1h:
        long_conds.append("MACD(1H) 골든크로스 발생")
    if price <= bb_lower * 1.003:
        long_conds.append(f"BB 하단 터치 (${price:,.0f} ≈ ${bb_lower:,.0f})")
    if rsi_4h < 45 and macd_4h > sig_4h:
        long_conds.append(f"4H 반등 구간 (RSI {rsi_4h:.1f} + MACD 상향)")
    if price > ema21_1h and price > ema50_1h and rsi_4h < 55 and adx_4h > 20:
        long_conds.append("EMA 위 안착 + 4H 추세 상승")

    # ── 숏 조건 ──
    if rsi_1h > 72:
        short_conds.append(f"RSI(1H) 과매수 {rsi_1h:.1f} (>72)")
    if macd_prev > sig_prev and macd_1h < sig_1h:
        short_conds.append("MACD(1H) 데드크로스 발생")
    if price >= bb_upper * 0.997:
        short_conds.append(f"BB 상단 터치 (${price:,.0f} ≈ ${bb_upper:,.0f})")
    if rsi_4h > 65 and macd_4h < sig_4h:
        short_conds.append(f"4H 하락 전환 (RSI {rsi_4h:.1f} + MACD 하향)")
    if price < ema21_1h and price < ema50_1h and rsi_4h > 50 and adx_4h > 20:
        short_conds.append("EMA 아래 이탈 + 4H 추세 하락")

    signals = []
    if len(long_conds) >= 2:
        signals.append(("LONG",  long_conds,  price))
    if len(short_conds) >= 2:
        signals.append(("SHORT", short_conds, price))

    return signals, indicators


# ─── 포지션 계산 ────────────────────────────────────────────────────────────────

def calc_position(signal_type: str, price: float) -> dict:
    stop_pct = 0.025  # 2.5% 손절
    if signal_type == "LONG":
        stop  = price * (1 - stop_pct)
        tp1   = price * 1.040
        tp2   = price * 1.085
    else:
        stop  = price * (1 + stop_pct)
        tp1   = price * 0.960
        tp2   = price * 0.915

    risk_usd     = TOTAL_CAPITAL * RISK_PCT                       # 위험금액 ($47.56)
    risk_per_unit = abs(price - stop)
    raw_pos      = (risk_usd / risk_per_unit) * price             # 리스크 기반 포지션
    position_usd = min(raw_pos, TOTAL_CAPITAL * MAX_POSITION_PCT) # 최대 30% 캡
    contracts    = position_usd / price

    return {
        "entry":        price,
        "stop":         stop,
        "tp1":          tp1,
        "tp2":          tp2,
        "leverage":     DEFAULT_LEVERAGE,
        "position_usd": position_usd,
        "contracts":    contracts,
        "risk_usd":     risk_usd,
        "stop_pct":     stop_pct * 100,
        "tp1_pct":      abs(tp1 - price) / price * 100,
        "tp2_pct":      abs(tp2 - price) / price * 100,
        "rr_ratio":     abs(tp1 - price) / abs(price - stop),
    }


# ─── 메시지 포맷 ────────────────────────────────────────────────────────────────

def format_alert(signal_type: str, conditions: list, ind: dict, pos: dict) -> str:
    emoji  = "🟢" if signal_type == "LONG" else "🔴"
    action = "롱(매수)" if signal_type == "LONG" else "숏(매도)"
    now    = datetime.now().strftime('%Y-%m-%d %H:%M')

    return f"""⚡ BTC 진입 신호!
{emoji} {action} 조건 충족
{now} KST
━━━━━━━━━━━━━━━━━━━━━━

💰 현재가: ${ind['price']:,.2f}

✅ 트리거 조건 ({len(conditions)}개):
{chr(10).join('  • ' + c for c in conditions)}

📊 보조 지표:
  RSI 1H: {ind['rsi_1h']:.1f} | RSI 4H: {ind['rsi_4h']:.1f}
  MACD 1H: {'상향' if ind['macd_1h'] > ind['sig_1h'] else '하향'}
  MACD 4H: {'상향' if ind['macd_4h'] > ind['sig_4h'] else '하향'}
  ADX 4H: {ind['adx_4h']:.1f} ({'강추세' if ind['adx_4h'] > 25 else '추세형성중'})
  BB: ${ind['bb_lower']:,.0f} ~ ${ind['bb_upper']:,.0f}

━━━━━━━━━━━━━━━━━━━━━━
💼 추천 포지션 (보유: ${TOTAL_CAPITAL:,} USDT)

  진입가:   ${pos['entry']:,.2f}
  손절가:   ${pos['stop']:,.2f}  (-{pos['stop_pct']:.1f}%)
  목표가1:  ${pos['tp1']:,.2f}  (+{pos['tp1_pct']:.1f}%)
  목표가2:  ${pos['tp2']:,.2f}  (+{pos['tp2_pct']:.1f}%)
  손익비:   1 : {pos['rr_ratio']:.1f}

  투입금액: ${pos['position_usd']:,.0f} (자본의 {pos['position_usd']/TOTAL_CAPITAL*100:.0f}%)
  레버리지: {pos['leverage']}x
  계약수량: {pos['contracts']:.4f} BTC
  최대손실: ${pos['risk_usd']:.1f} (자본의 {RISK_PCT*100:.0f}%)

💡 분할 진입 권장
  1차: 50% 즉시 진입 (${pos['position_usd']*0.5:,.0f})
  2차: 50% 확인 후 진입 (${pos['position_usd']*0.5:,.0f})

⚠️ 참고용이며 투자 책임은 본인에게 있습니다."""


# ─── 텔레그램 ────────────────────────────────────────────────────────────────────

def send_telegram(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    return r.ok


# ─── 메인 ────────────────────────────────────────────────────────────────────────

def main():
    now_str = datetime.now().strftime('%H:%M:%S')
    print(f"[{now_str}] BTC 진입 조건 체크 중...")

    state = load_state()

    try:
        df_1h, df_4h = fetch_data()
    except Exception as e:
        print(f"  데이터 수집 실패: {e}")
        return

    signals, indicators = check_signals(df_1h, df_4h)
    print(f"  현재가: ${indicators['price']:,.0f} | RSI 1H: {indicators['rsi_1h']:.1f} | RSI 4H: {indicators['rsi_4h']:.1f}")

    if not signals:
        print("  신호 없음")
        return

    for signal_type, conditions, price in signals:
        key = f"last_{signal_type.lower()}"
        last_time = state.get(key)

        if last_time:
            elapsed_h = (datetime.now() - datetime.fromisoformat(last_time)).total_seconds() / 3600
            if elapsed_h < COOLDOWN_HOURS:
                print(f"  {signal_type} 쿨다운 중 ({elapsed_h:.1f}h/{COOLDOWN_HOURS}h)")
                continue

        pos   = calc_position(signal_type, price)
        msg   = format_alert(signal_type, conditions, indicators, pos)

        if send_telegram(msg):
            state[key] = datetime.now().isoformat()
            save_state(state)
            print(f"  ✅ {signal_type} 알림 전송!")
        else:
            print(f"  ❌ 텔레그램 전송 실패")


if __name__ == "__main__":
    main()
