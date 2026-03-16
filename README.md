# 📊 Stock Analysis PRO

A Bloomberg-style terminal for stock analysis, built with Flask and deployed on Render.

🔗 **Live:** [stockmaxing.onrender.com](https://stockmaxing.onrender.com)

---

## Features

### Stock Analysis
- **Technical** — RSI, MACD, Bollinger Bands, SMA 20/50/200, volume analysis
- **Fundamental** — P/E, Forward P/E, EV/EBITDA, margins, ROE, ROA, debt/equity, free cash flow
- **Risk** — Beta, Sharpe ratio, Value at Risk (95%), max drawdown, annualised volatility
- **Charts** — Interactive TradingView chart (with Volume by Price) + static matplotlib chart
- **News** — Latest headlines with AI sentiment tagging (POSITIVE / NEUTRAL / NEGATIVE)
- **Earnings** — Next earnings date + EPS beat/miss history
- **Analysts** — Wall Street consensus, price targets (low/mean/high), upgrades & downgrades
- **Insider** — SEC Form 4 filings (executives buying and selling)
- **AI Analysis** — LLaMA 3.3 70B via Groq: overall assessment, strengths, risks, verdict

### Other Views
- **Watchlist** — Track tickers with live prices, sparklines, and daily/weekly/monthly returns
- **Portfolio** — Track positions with live P&L, sector breakdown, and AI portfolio analysis
- **Compare** — Side-by-side comparison of multiple stocks with AI ranking
- **Sectors** — S&P 500 sector overview via ETFs, sorted by daily performance, with top holdings

### Top Bar
- Live macro ticker: S&P 500, Nasdaq, VIX, US 10Y Yield, Gold, Oil, EUR/USD, USD/NOK
- US market open/closed status and clock
- Dark/light mode toggle

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask, Gunicorn |
| Data | yfinance (Yahoo Finance) |
| AI | Groq API — LLaMA 3.3 70B Versatile |
| Charts | TradingView Widget, Matplotlib, Seaborn |
| Frontend | Vanilla JS, IBM Plex Mono, CSS Grid |
| Hosting | Render (free tier) |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyse` | POST | Market data, technicals, fundamentals, risk |
| `/api/ai_analyse` | POST | AI analysis (runs async after main analyse) |
| `/api/makro` | GET | Live macro indicators for top bar |
| `/api/earnings` | POST | Next earnings date + EPS history |
| `/api/analyst` | POST | Analyst ratings, price targets, upgrades |
| `/api/insider` | POST | Insider transactions (SEC Form 4) |
| `/api/sammenlign` | POST | Multi-stock comparison with AI ranking |
| `/api/portefolje_analyse` | POST | Portfolio analysis with AI |
| `/api/nyheter` | POST | News with AI sentiment |
| `/api/watchlist_kurs` | POST | Watchlist prices and sparklines |

---

## Setup & Deployment

### Local development
```bash
pip install -r requirements.txt
python app.py
```

### Deploy to Render
1. Fork this repo
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect your GitHub repo
4. Add environment variable:
   ```
   GROQ_API_KEY = your_key_here
   ```
5. Render auto-deploys on every push to `main`

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys)

---

## Project Structure

```
STOCKMAXING/
├── app.py               # Flask backend + all API endpoints
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment config
└── templates/
    └── index.html       # Full frontend (HTML + CSS + JS)
```

---

## Notes

- Data is sourced from Yahoo Finance via `yfinance` — availability varies by ticker
- Norwegian stocks use `.OL` suffix (e.g. `EQNR.OL`, `DNB.OL`)
- AI analysis loads asynchronously after market data — no waiting for results
- All user data (watchlist, portfolio) is stored in browser `localStorage`
- **Not financial advice**

---

*Built by Sebastian Johnsen*
