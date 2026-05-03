import os
import json
import warnings
import requests
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

COOLDOWN_HOURS  = 8    # 같은 종목 재알림 방지
ENTRY_TOLERANCE = 0.02 # 진입가 ±2% 이내면 알림

IS_GHA     = bool(os.getenv("GITHUB_ACTIONS"))
STATE_FILE = Path("stock_alert_state.json") if IS_GHA else Path.home() / ".stock_alert_state.json"

# 추천 종목 캐시 (매일 리포트 생성 시 업데이트됨)
WATCHLIST_FILE = Path("stock_watchlist.json") if IS_GHA else Path.home() / ".stock_watchlist.json"


# ─── 상태 / 워치리스트 관리 ────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}

def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, default=str, indent=2))


def save_watchlist(ranked: list):
    """매일 리포트 생성 후 호출 — 진입 후보 저장"""
    watchlist = {}
    for ticker, tech, fund in ranked:
        price = tech['price']
        watchlist[ticker] = {
            "name":        fund.get("name", ticker),
            "sector":      fund.get("sector", "N/A"),
            "entry_low":   round(price * 0.97, 2),   # 현재가 -3%
            "entry_high":  round(price * 1.005, 2),  # 현재가 +0.5%
            "target":      fund.get("analyst_target") or round(price * 1.15, 2),
            "stop":        round(price * 0.93, 2),   # -7% 손절
            "saved_at":    datetime.now().isoformat(),
        }
    save_json(WATCHLIST_FILE, watchlist)
    print(f"  워치리스트 저장: {list(watchlist.keys())}")


# ─── 실시간 가격 체크 ─────────────────────────────────────────────────────────

def fetch_current_prices(tickers: list) -> dict:
    if not tickers:
        return {}
    try:
        raw = yf.download(tickers, period="5d", interval="1h",
                          progress=False, auto_adjust=True, group_by="ticker")
        prices = {}
        for ticker in tickers:
            try:
                df = raw[ticker] if len(tickers) > 1 else raw
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df.columns = [c.lower() for c in df.columns]
                prices[ticker] = float(df["close"].dropna().iloc[-1])
            except Exception:
                pass
        return prices
    except Exception:
        return {}


def fetch_rsi(ticker: str) -> float | None:
    try:
        df = yf.download(ticker, period="30d", interval="1h",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [c.lower() for c in df.columns]
        rsi = ta.momentum.RSIIndicator(df["close"].dropna(), window=14).rsi()
        return float(rsi.iloc[-1])
    except Exception:
        return None


# ─── 진입 조건 체크 ──────────────────────────────────────────────────────────

def check_entry(ticker: str, current_price: float, watch: dict) -> tuple[bool, list]:
    conditions = []

    entry_low  = watch["entry_low"]
    entry_high = watch["entry_high"]
    target     = watch["target"]
    stop       = watch["stop"]

    # 1. 가격이 진입 구간 안에 있는지
    if entry_low <= current_price <= entry_high:
        conditions.append(f"진입 구간 도달 (${entry_low:,.2f} ~ ${entry_high:,.2f})")

    # 2. 저장 시점 대비 눌림 (2~5% 하락 = 매수 기회)
    try:
        saved_entry = (entry_low + entry_high) / 2
        pullback = (saved_entry - current_price) / saved_entry * 100
        if 2 <= pullback <= 7:
            conditions.append(f"눌림목 진입 기회 ({pullback:.1f}% 하락)")
    except Exception:
        pass

    # 3. RSI 보조 확인 (개별 요청 — 조건 충족 시만)
    if conditions:
        rsi = fetch_rsi(ticker)
        if rsi:
            if rsi < 45:
                conditions.append(f"RSI 과매도 구간 ({rsi:.1f})")
            elif rsi > 70:
                conditions = []  # 과매수면 취소
                return False, []

    # 손익비 체크 (최소 2:1 이상)
    if current_price > 0 and target and stop:
        rr = abs(target - current_price) / abs(current_price - stop)
        if rr < 1.5:
            return False, []

    triggered = len(conditions) >= 1
    return triggered, conditions


# ─── 메시지 포맷 ─────────────────────────────────────────────────────────────

def format_alert(ticker: str, watch: dict, current_price: float,
                 conditions: list) -> str:
    target   = watch["target"]
    stop     = watch["stop"]
    upside   = (target - current_price) / current_price * 100
    downside = (current_price - stop) / current_price * 100
    rr       = upside / downside if downside > 0 else 0

    return f"""📈 주식 진입 신호!
🟢 매수 조건 충족: {ticker}
{datetime.now().strftime('%Y-%m-%d %H:%M')} KST
━━━━━━━━━━━━━━━━━━━━━━

🏢 {watch.get('name', ticker)} ({watch.get('sector', '')})
💰 현재가:  ${current_price:,.2f}
━━━━━━━━━━━━━━━━━━━━━━

✅ 트리거 조건:
{chr(10).join('  • ' + c for c in conditions)}

📊 트레이딩 플랜:
  매수 구간:  ${watch['entry_low']:,.2f} ~ ${watch['entry_high']:,.2f}
  목표가:    ${target:,.2f}  (+{upside:.1f}%)
  손절가:    ${stop:,.2f}  (-{downside:.1f}%)
  손익비:    1 : {rr:.1f}

💡 매수 전략:
  1차 (50%): 현재가 ${current_price:,.2f} 즉시
  2차 (50%): ${watch['entry_low']:,.2f} 이하 추가 매수

⚠️ 참고용이며 투자 책임은 본인에게 있습니다."""


# ─── 텔레그램 ─────────────────────────────────────────────────────────────────

def send_telegram(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    return r.ok


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 주식 진입 조건 체크 중...")

    watchlist = load_json(WATCHLIST_FILE)
    if not watchlist:
        print("  워치리스트 없음 — 오늘의 주식 리포트가 아직 생성되지 않았습니다.")
        return

    state    = load_json(STATE_FILE)
    tickers  = list(watchlist.keys())
    prices   = fetch_current_prices(tickers)

    if not prices:
        print("  가격 데이터 수집 실패")
        return

    print(f"  모니터링 종목: {tickers}")

    for ticker, watch in watchlist.items():
        current_price = prices.get(ticker)
        if not current_price:
            continue

        triggered, conditions = check_entry(ticker, current_price, watch)

        if not triggered:
            continue

        # 쿨다운 체크
        last_time = state.get(ticker)
        if last_time:
            elapsed_h = (datetime.now() - datetime.fromisoformat(last_time)).total_seconds() / 3600
            if elapsed_h < COOLDOWN_HOURS:
                print(f"  {ticker} 쿨다운 중 ({elapsed_h:.1f}h/{COOLDOWN_HOURS}h)")
                continue

        msg = format_alert(ticker, watch, current_price, conditions)

        if send_telegram(msg):
            state[ticker] = datetime.now().isoformat()
            save_json(STATE_FILE, state)
            print(f"  ✅ {ticker} 알림 전송!")
        else:
            print(f"  ❌ {ticker} 텔레그램 전송 실패")


if __name__ == "__main__":
    main()
