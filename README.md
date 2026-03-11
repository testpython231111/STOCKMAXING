# 📈 Aksjeanalyse PRO — Webapp

Bloomberg-inspirert aksjeanalyse webapp med AI-vurdering.

## Funksjoner
- Teknisk analyse (RSI, MACD, Bollinger Bands, MA)
- Fundamental analyse (P/E, P/B, marginer, ROE)
- Risiko & avkastning (Sharpe, Beta, VaR, Drawdown)
- AI-vurdering via Groq (gratis)
- Makroøkonomiske indikatorer
- Fondssammenligning

## Kjør lokalt

```bash
pip install -r requirements.txt
python app.py
# Åpne http://localhost:5000
```

## Deploy til Render.com (gratis, online for alle)

1. Lag konto på [github.com](https://github.com) og last opp disse filene
2. Lag konto på [render.com](https://render.com)
3. Klikk **New → Web Service**
4. Koble til GitHub-repoet ditt
5. Render oppdager `render.yaml` automatisk
6. Klikk **Deploy** — ferdig om ~3 minutter!

URL-en din blir: `https://aksjeanalyse-pro.onrender.com`

## Groq API-nøkkel
Hent gratis på [console.groq.com/keys](https://console.groq.com/keys).
Limes inn direkte i webappen — lagres i nettleseren.
