#!/bin/bash
set -e

echo "📦 BTC 리포트 환경 설정 중..."

# Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ 설치 완료!"
echo ""
echo "⚠️  .env 파일을 열어 ANTHROPIC_API_KEY를 입력해주세요:"
echo "   nano /Users/henry/btc_report/.env"
echo ""
echo "🚀 테스트 실행:"
echo "   cd /Users/henry/btc_report && source venv/bin/activate && python main.py"
echo ""
echo "⏰ 매일 자동 실행 등록 (오전 9시):"
echo "   crontab -e"
echo "   추가할 내용:"
echo "   0 9 * * * cd /Users/henry/btc_report && source venv/bin/activate && python main.py >> /Users/henry/btc_report/cron.log 2>&1"
