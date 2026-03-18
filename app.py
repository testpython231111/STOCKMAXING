"""
Aksjeanalyse PRO — Flask Web App
"""

import os, io, base64, warnings, json, gc, time, logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from flask import Flask, request, jsonify, render_template
import requests as _requests

warnings.filterwarnings("ignore")

app = Flask(__name__)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RISIKOFRI_RENTE = 0.045
BENCHMARK       = "^GSPC"

MAKRO_TICKERS = {
    "S&P 500":       "^GSPC",
    "Nasdaq":        "^IXIC",
    "VIX":           "^VIX",
    "US 10y Yield":  "^TNX",
    "Gold":          "GC=F",
    "Oil (Brent)":  "BZ=F",
    "EUR/USD":       "EURUSD=X",
    "USD/NOK":       "USDNOK=X",
}

# ── Simple in-memory cache (TTL-based) ───────────────────────────────────────
_cache = {}
_cache_lock = Lock()
CACHE_TTL = 300  # 5 minutes

def cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() - entry["ts"] < CACHE_TTL:
            return entry["data"]
        return None

def cache_set(key, data):
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}
        # Evict old entries if cache grows large
        if len(_cache) > 200:
            cutoff = time.time() - CACHE_TTL
            for k in [k for k,v in _cache.items() if v["ts"] < cutoff]:
                del _cache[k]

# ── Hjelpefunksjoner ──────────────────────────────────────────────────────────

# Flask 3.x compatible NaN sanitizer — replaces NaN/Inf with None before jsonify
def sanitize(obj):
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    if isinstance(obj, dict):  return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [sanitize(v) for v in obj]
    return obj

def safe_jsonify(data):
    return jsonify(sanitize(data))

def safe(v, pst=False, dec=2):
    try:
        if v is None or (isinstance(v, float) and np.isnan(float(v))): return "N/A"
        return f"{float(v)*100:.{dec}f}%" if pst else f"{float(v):.{dec}f}"
    except: return "N/A"

def stor_tall(n):
    try:
        n = float(n)
        for g, e in [(1e12,"T"),(1e9,"B"),(1e6,"M")]:
            if abs(n) >= g: return f"{n/g:.2f}{e}"
        return f"{n:.2f}"
    except: return "N/A"

def spør_ai(prompt: str, api_key: str, maks=1200) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key: return "No API key configured."
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": maks,
                "temperature": 0.7,
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }
        r = _requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        logger.error(f"[AI] API error: {e}")
        return f"AI error: {e}"

# ── Teknisk analyse ───────────────────────────────────────────────────────────

def beregn_tekniske(df):
    c = df["Close"]
    df["SMA20"]  = c.rolling(20).mean()
    df["SMA50"]  = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["EMA12"]  = c.ewm(span=12, adjust=False).mean()
    df["EMA26"]  = c.ewm(span=26, adjust=False).mean()
    df["MACD"]   = df["EMA12"] - df["EMA26"]
    df["MACD_S"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_H"] = df["MACD"] - df["MACD_S"]
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    df["BB_Mid"] = c.rolling(20).mean()
    std          = c.rolling(20).std()
    df["BB_Up"]  = df["BB_Mid"] + 2*std
    df["BB_Lo"]  = df["BB_Mid"] - 2*std
    df["BB_pct"] = (c - df["BB_Lo"]) / (df["BB_Up"] - df["BB_Lo"])
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-c.shift()).abs(),
                    (df["Low"]-c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"]       = tr.rolling(14).mean()
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    return df

def hent_signaler(df):
    s = df.iloc[-1]
    p = float(s["Close"])
    return {
        "pris":       round(p, 2),
        "rsi":        round(float(s["RSI"]),    1) if not np.isnan(s["RSI"])    else 50,
        "macd_bull":  bool(s["MACD"] > s["MACD_S"]),
        "over_sma20": bool(p > s["SMA20"])  if not np.isnan(s["SMA20"])  else False,
        "over_sma50": bool(p > s["SMA50"])  if not np.isnan(s["SMA50"])  else False,
        "over_sma200":bool(p > s["SMA200"]) if not np.isnan(s["SMA200"]) else False,
        "bb_pct":     round(float(s["BB_pct"]),3) if not np.isnan(s["BB_pct"]) else 0.5,
        "sma20":      round(float(s["SMA20"]),  2) if not np.isnan(s["SMA20"])  else p,
        "sma50":      round(float(s["SMA50"]),  2) if not np.isnan(s["SMA50"])  else p,
        "sma200":     round(float(s["SMA200"]), 2) if not np.isnan(s["SMA200"]) else p,
        "bb_upper":   round(float(s["BB_Up"]),  2) if not np.isnan(s["BB_Up"])  else p,
        "bb_lower":   round(float(s["BB_Lo"]),  2) if not np.isnan(s["BB_Lo"])  else p,
        "atr":        round(float(s["ATR"]),    2) if not np.isnan(s["ATR"])    else 0,
    }

# ── Risiko ────────────────────────────────────────────────────────────────────

def beregn_risiko(df, bm_df):
    ret  = df["Close"].pct_change().dropna()
    bret = bm_df["Close"].pct_change().dropna()
    felles = ret.index.intersection(bret.index)
    ret, bret = ret.loc[felles], bret.loc[felles]
    if len(ret) < 10: return {}
    hd = 252
    total  = (df["Close"].iloc[-1]/df["Close"].iloc[0]-1)*100
    dager  = (df.index[-1]-df.index[0]).days
    cagr   = ((df["Close"].iloc[-1]/df["Close"].iloc[0])**(365/max(dager,1))-1)*100
    dagvol = ret.std()
    årvol  = dagvol*np.sqrt(hd)*100
    rf     = RISIKOFRI_RENTE/hd
    sharpe = (ret.mean()-rf)/dagvol*np.sqrt(hd)
    neg    = ret[ret < rf]
    dside  = neg.std()*np.sqrt(hd) if len(neg)>1 else float("nan")
    sortino = (ret.mean()*hd-RISIKOFRI_RENTE)/dside if dside and dside!=0 else float("nan")
    rm     = df["Close"].cummax()
    dd     = (df["Close"]-rm)/rm
    max_dd = dd.min()*100
    kov    = np.cov(ret, bret)
    beta   = kov[0,1]/kov[1,1] if kov[1,1]!=0 else float("nan")
    alpha  = (ret.mean()-(rf+beta*(bret.mean()-rf)))*hd*100
    var95  = np.percentile(ret,5)*100
    var99  = np.percentile(ret,1)*100
    calmar = cagr/abs(max_dd) if max_dd!=0 else float("nan")
    return dict(total=round(total,2), cagr=round(cagr,2), vol=round(årvol,2),
                sharpe=round(sharpe,3), sortino=round(sortino,3) if not np.isnan(sortino) else None,
                calmar=round(calmar,3) if not np.isnan(calmar) else None,
                max_dd=round(max_dd,2), beta=round(beta,3) if not np.isnan(beta) else None,
                alpha=round(alpha,2), var95=round(var95,2), var99=round(var99,2),
                dd_series=dd.tolist(), dd_dates=[str(d)[:10] for d in dd.index])

# ── Graf → base64 ─────────────────────────────────────────────────────────────

def lag_graf(df: pd.DataFrame, ticker: str) -> str:
    style = {
        "axes.facecolor":  "#06090F",
        "figure.facecolor":"#06090F",
        "axes.edgecolor":  "#1A2D45",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color":     "#7A92B0",
        "ytick.color":     "#7A92B0",
        "text.color":      "#7A92B0",
        "grid.color":      "#1A2D45",
        "grid.linestyle":  "--",
        "grid.alpha":      0.3,
    }
    with matplotlib.rc_context(style):
        fig = plt.figure(figsize=(16, 12), facecolor="#06090F")
        gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.3)
        C   = {"kurs":"#4A9ED6","sma20":"#F0B429","sma50":"#64B5F6",
               "sma200":"#E05470","bull":"#4A9ED6","bear":"#E05470","text":"#7A92B0"}

        ax1 = fig.add_subplot(gs[0,:])
        ax1.set_facecolor("#06090F")
        ax1.plot(df.index, df["Close"],  color=C["kurs"],  lw=1.5, label="Kurs")
        ax1.plot(df.index, df["SMA20"],  color=C["sma20"], lw=0.8, ls="--", label="SMA20")
        ax1.plot(df.index, df["SMA50"],  color=C["sma50"], lw=0.8, ls="--", label="SMA50")
        ax1.plot(df.index, df["SMA200"], color=C["sma200"],lw=0.8, ls="--", label="SMA200")
        ax1.fill_between(df.index, df["BB_Up"], df["BB_Lo"],
                         alpha=0.05, color=C["kurs"])
        ax1.set_title(f"{ticker} — Kurs & Indikatorer", color=C["text"], fontsize=10)
        ax1.legend(fontsize=7, labelcolor=C["text"])
        ax1.tick_params(colors=C["text"]); ax1.spines[:].set_color("#1A2D45")

        ax2 = fig.add_subplot(gs[1,:])
        ax2.set_facecolor("#06090F")
        fc = [C["bull"] if df["Close"].iloc[i]>=df["Close"].iloc[i-1] else C["bear"]
              for i in range(1,len(df))]
        ax2.bar(df.index[1:], df["Volume"].iloc[1:], color=fc, alpha=0.6, width=1)
        ax2.plot(df.index, df["Vol_SMA20"], color=C["sma20"], lw=0.8, ls="--")
        ax2.set_title("Volume", color=C["text"], fontsize=10)
        ax2.tick_params(colors=C["text"]); ax2.spines[:].set_color("#1A2D45")

        ax3 = fig.add_subplot(gs[2,0])
        ax3.set_facecolor("#06090F")
        ax3.plot(df.index, df["RSI"], color="#f4a261", lw=1.2)
        ax3.axhline(70, color=C["bear"], ls="--", lw=0.7, alpha=0.6)
        ax3.axhline(30, color=C["bull"], ls="--", lw=0.7, alpha=0.6)
        ax3.fill_between(df.index, df["RSI"], 70, where=df["RSI"]>=70, alpha=0.1, color=C["bear"])
        ax3.fill_between(df.index, df["RSI"], 30, where=df["RSI"]<=30, alpha=0.1, color=C["bull"])
        ax3.set_ylim(0,100); ax3.set_title("RSI (14d)", color=C["text"], fontsize=10)
        ax3.tick_params(colors=C["text"]); ax3.spines[:].set_color("#1A2D45")

        ax4 = fig.add_subplot(gs[2,1])
        ax4.set_facecolor("#06090F")
        ax4.plot(df.index, df["MACD"],   color=C["sma50"], lw=1.2, label="MACD")
        ax4.plot(df.index, df["MACD_S"], color=C["sma20"], lw=1.0, ls="--", label="Signal")
        hc = [C["bull"] if v>=0 else C["bear"] for v in df["MACD_H"]]
        ax4.bar(df.index, df["MACD_H"], color=hc, alpha=0.5, width=1)
        ax4.set_title("MACD", color=C["text"], fontsize=10)
        ax4.legend(fontsize=7, labelcolor=C["text"])
        ax4.tick_params(colors=C["text"]); ax4.spines[:].set_color("#1A2D45")

        ax5 = fig.add_subplot(gs[3,0])
        ax5.set_facecolor("#06090F")
        rm = df["Close"].cummax()
        dd = (df["Close"]-rm)/rm*100
        ax5.fill_between(df.index, dd, 0, color=C["bear"], alpha=0.5)
        ax5.plot(df.index, dd, color=C["bear"], lw=0.8)
        ax5.set_title("Drawdown (%)", color=C["text"], fontsize=10)
        ax5.tick_params(colors=C["text"]); ax5.spines[:].set_color("#1A2D45")

        ax6 = fig.add_subplot(gs[3,1])
        ax6.set_facecolor("#06090F")
        dagret = df["Close"].pct_change().dropna()*100
        ax6.hist(dagret, bins=50, color=C["sma50"], alpha=0.6, edgecolor="none")
        ax6.axvline(dagret.mean(), color=C["sma20"], lw=1.2, ls="--")
        ax6.axvline(np.percentile(dagret,5), color=C["bear"], lw=1.0, ls=":")
        ax6.set_title("Return Distribution", color=C["text"], fontsize=10)
        ax6.tick_params(colors=C["text"]); ax6.spines[:].set_color("#1A2D45")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#06090F")
        plt.close(fig)
        gc.collect()
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# ── Flask ruter ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data   = request.json
    ticker = data.get("ticker","").strip().upper()
    periode= data.get("periode","1y")

    if not ticker:
        return safe_jsonify({"error": "No ticker provided"}), 400

    # Check cache
    cache_key = f"analyse:{ticker}:{periode}"
    cached = cache_get(cache_key)
    if cached:
        return safe_jsonify(cached)

    try:
        # Fetch stock data + benchmark in parallel
        def fetch_stock():
            aksje = yf.Ticker(ticker)
            df = yf.download(ticker, period=periode, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return aksje, df

        def fetch_benchmark():
            try:
                bm = yf.download(BENCHMARK, period=periode, progress=False, auto_adjust=True)
                if isinstance(bm.columns, pd.MultiIndex): bm.columns = bm.columns.get_level_values(0)
                return bm
            except: return None

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_stock = ex.submit(fetch_stock)
            f_bm    = ex.submit(fetch_benchmark)
            aksje, df = f_stock.result()
            bm_result = f_bm.result()

        if df.empty:
            return safe_jsonify({"error": f"No data for {ticker}. Check the ticker symbol."}), 404

        info  = aksje.info
        bm_df = bm_result if bm_result is not None and not bm_result.empty else df.copy()

    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500

    df  = beregn_tekniske(df)
    sig = hent_signaler(df)
    ri  = beregn_risiko(df, bm_df)

    # Kurshistorikk for chart
    historikk = {
        "datoer": [str(d)[:10] for d in df.index],
        "priser": [round(float(p),2) for p in df["Close"]],
    }

    fundamental = {
        "navn":       info.get("longName", ticker),
        "sektor":     info.get("sector","N/A"),
        "bransje":    info.get("industry","N/A"),
        "børs":       info.get("exchange","N/A"),
        "valuta":     info.get("currency","N/A"),
        "pris":       safe(info.get("currentPrice")),
        "pe":         safe(info.get("trailingPE")),
        "forward_pe": safe(info.get("forwardPE")),
        "pb":         safe(info.get("priceToBook")),
        "ps":         safe(info.get("priceToSalesTrailing12Months")),
        "eps":        safe(info.get("trailingEps")),
        "mktcap":     stor_tall(info.get("marketCap")),
        "omsetning":  stor_tall(info.get("totalRevenue")),
        "ebitda":     stor_tall(info.get("ebitda")),
        "frikontant": stor_tall(info.get("freeCashflow")),
        "gjeld_ek":   safe(info.get("debtToEquity")),
        "utbytte":    safe(info.get("dividendYield"), pst=True),
        "roe":        safe(info.get("returnOnEquity"), pst=True),
        "roa":        safe(info.get("returnOnAssets"), pst=True),
        "bruttomargin":  safe(info.get("grossMargins"), pst=True),
        "driftsmargin":  safe(info.get("operatingMargins"), pst=True),
        "nettmargin":    safe(info.get("profitMargins"), pst=True),
        "52u_høy":    safe(info.get("fiftyTwoWeekHigh")),
        "52u_lav":    safe(info.get("fiftyTwoWeekLow")),
        "mål":        safe(info.get("targetMeanPrice")),
        "konsensus":  info.get("recommendationKey","N/A").upper(),
        # Raw numeric fields for Margin of Safety calculations (None if unavailable)
        "_pris_raw":          info.get("currentPrice") or info.get("regularMarketPrice"),
        "_pe_raw":            info.get("trailingPE"),
        "_eps_raw":           info.get("trailingEps"),
        "_pb_raw":            info.get("priceToBook"),
        "_earnings_growth":   info.get("earningsGrowth"),   # trailing YoY EPS growth
        "_revenue_growth":    info.get("revenueGrowth"),
        "_52u_høy_raw":       info.get("fiftyTwoWeekHigh"),
        "_52u_lav_raw":       info.get("fiftyTwoWeekLow"),
        "_mål_raw":           info.get("targetMeanPrice"),
        "_evEbitda_raw":      info.get("enterpriseToEbitda"),
        "_peg_raw":           info.get("pegRatio"),
    }

    result = {
        "ticker":       ticker,
        "fundamental":  fundamental,
        "signaler":     sig,
        "risiko":       ri,
        "graf":         "",  # loaded separately via /api/chart
        "historikk":    historikk,
        "ai":           "",
    }
    cache_set(cache_key, result)
    return safe_jsonify(result)


@app.route("/api/chart", methods=["POST"])
def api_chart():
    data    = request.json
    ticker  = data.get("ticker","").strip().upper()
    periode = data.get("periode","1y")
    if not ticker:
        return safe_jsonify({"error": "No ticker"}), 400
    cache_key = f"chart:{ticker}:{periode}"
    cached = cache_get(cache_key)
    if cached:
        return safe_jsonify({"graf": cached})
    try:
        df = yf.download(ticker, period=periode, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty:
            return safe_jsonify({"error": "No data"}), 404
        df = beregn_tekniske(df)
        graf = lag_graf(df, ticker)
        cache_set(cache_key, graf)
        return safe_jsonify({"graf": graf})
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500

@app.route("/api/dcf", methods=["POST"])
def api_dcf():
    """
    Professional DCF valuation:
      • WACC built from CAPM (cost of equity) + after-tax cost of debt
      • 2-stage DCF: 5-year explicit FCF projection + Gordon Growth terminal value
      • Bull / Base / Bear scenarios
      • Sensitivity matrix: intrinsic value across growth × WACC grid
    """
    data   = request.json
    ticker = data.get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker"}), 400

    cache_key = f"dcf:{ticker}"
    cached = cache_get(cache_key)
    if cached:
        return safe_jsonify(cached)

    try:
        aksje = yf.Ticker(ticker)
        info  = aksje.info

        # ── Raw inputs ──────────────────────────────────────────────────────
        cur_price     = info.get("currentPrice") or info.get("regularMarketPrice")
        shares_out    = info.get("sharesOutstanding")
        market_cap    = info.get("marketCap")
        total_debt    = info.get("totalDebt", 0) or 0
        cash          = info.get("totalCash", 0) or 0
        fcf           = info.get("freeCashflow")
        revenue       = info.get("totalRevenue")
        ebitda_val    = info.get("ebitda")
        op_margin     = info.get("operatingMargins")
        beta          = info.get("beta", 1.0) or 1.0
        tax_rate      = info.get("effectiveTaxRate", 0.21) or 0.21
        interest_exp  = abs(info.get("interestExpense", 0) or 0)
        sector        = info.get("sector", "")
        currency      = info.get("currency", "USD")

        # Validate we have the minimum needed
        if not fcf or not shares_out or not cur_price:
            return safe_jsonify({"error": "Insufficient financial data for DCF (need FCF, shares outstanding, current price)."}), 422

        # ── WACC ────────────────────────────────────────────────────────────
        RF          = RISIKOFRI_RENTE           # 4.5 % — matches global constant
        ERP         = 0.055                     # equity risk premium 5.5 %
        cost_equity = RF + float(beta) * ERP    # CAPM

        enterprise_value = market_cap + total_debt - cash if market_cap else None
        debt_weight   = total_debt / (total_debt + market_cap) if market_cap and total_debt else 0.0
        equity_weight = 1.0 - debt_weight
        cost_debt_pre = (interest_exp / total_debt) if total_debt and interest_exp else 0.04
        cost_debt_pre = max(0.02, min(cost_debt_pre, 0.15))   # cap at reasonable range
        cost_debt     = cost_debt_pre * (1 - float(tax_rate))
        wacc          = equity_weight * cost_equity + debt_weight * cost_debt

        # ── Scenario growth assumptions ─────────────────────────────────────
        # Sector-aware base growth heuristics
        SECTOR_GROWTH = {
            "Technology": 0.12, "Healthcare": 0.09, "Communication Services": 0.10,
            "Consumer Discretionary": 0.08, "Financials": 0.07, "Industrials": 0.07,
            "Consumer Staples": 0.05, "Energy": 0.05, "Materials": 0.05,
            "Real Estate": 0.05, "Utilities": 0.04,
        }
        base_g = SECTOR_GROWTH.get(sector, 0.07)

        scenarios = {
            "Bear": {"g5": base_g - 0.05, "terminal": 0.020, "wacc_adj": +0.015, "label": "BEAR"},
            "Base": {"g5": base_g,        "terminal": 0.030, "wacc_adj":  0.000, "label": "BASE"},
            "Bull": {"g5": base_g + 0.05, "terminal": 0.040, "wacc_adj": -0.010, "label": "BULL"},
        }

        def run_dcf(fcf_base, g5, terminal_g, w):
            """Returns intrinsic equity value per share."""
            w = max(w, 0.03)  # floor WACC at 3 %
            g5 = max(g5, -0.20)
            terminal_g = min(terminal_g, w - 0.005)  # terminal g must be < WACC

            # Stage 1: explicit 5-year FCFs
            pv_fcfs = 0.0
            cf = float(fcf_base)
            for yr in range(1, 6):
                cf *= (1 + g5)
                pv_fcfs += cf / (1 + w) ** yr

            # Stage 2: terminal value (Gordon Growth) at end of year 5
            terminal_fcf = cf * (1 + terminal_g)
            terminal_val = terminal_fcf / (w - terminal_g)
            pv_terminal  = terminal_val / (1 + w) ** 5

            # Equity value = enterprise DCF − net debt → per share
            enterprise_dcf = pv_fcfs + pv_terminal
            equity_val     = enterprise_dcf + float(cash) - float(total_debt)
            equity_val     = max(equity_val, 0)
            return round(equity_val / float(shares_out), 2), round(pv_fcfs, 0), round(pv_terminal, 0)

        scenario_results = {}
        for name, sc in scenarios.items():
            w_adj    = wacc + sc["wacc_adj"]
            iv, pv1, pv2 = run_dcf(fcf, sc["g5"], sc["terminal"], w_adj)
            upside   = round((iv - float(cur_price)) / float(cur_price) * 100, 1) if cur_price else None
            scenario_results[name] = {
                "label":       sc["label"],
                "g5":          round(sc["g5"] * 100, 1),
                "terminal_g":  round(sc["terminal"] * 100, 1),
                "wacc":        round(w_adj * 100, 2),
                "intrinsic":   iv,
                "upside":      upside,
                "pv_fcfs":     pv1,
                "pv_terminal": pv2,
            }

        # ── Sensitivity matrix ──────────────────────────────────────────────
        # Rows = growth rate (stage 1), Cols = WACC
        g_range    = [base_g + d for d in (-0.08, -0.05, -0.02, 0, +0.03, +0.06, +0.09)]
        wacc_range = [wacc + d   for d in (-0.03, -0.02, -0.01, 0, +0.01, +0.02, +0.03)]
        base_terminal = scenarios["Base"]["terminal"]

        matrix = []
        for g in g_range:
            row = []
            for w in wacc_range:
                iv, _, _ = run_dcf(fcf, g, base_terminal, w)
                row.append(iv)
            matrix.append(row)

        # ── Explicit 5-year FCF table (Base scenario) ───────────────────────
        base_w = wacc + scenarios["Base"]["wacc_adj"]
        base_g5 = scenarios["Base"]["g5"]
        fcf_table = []
        cf = float(fcf)
        for yr in range(1, 6):
            cf *= (1 + base_g5)
            pv = cf / (1 + base_w) ** yr
            fcf_table.append({
                "year":    yr,
                "fcf":     round(cf / 1e9, 3),
                "pv":      round(pv / 1e9, 3),
            })

        # ── Margin of Safety indicators ──────────────────────────────────────
        mos_signals = []

        # 1. DCF Margin of Safety (Base case)
        dcf_base_iv   = scenario_results["Base"]["intrinsic"]
        dcf_base_up   = scenario_results["Base"]["upside"]
        if dcf_base_iv and dcf_base_up is not None:
            if dcf_base_up >= 30:
                dcf_verdict = ("STRONG BUY", "green", f"DCF intrinsic {dcf_base_up:+.1f}% above current price — significant margin of safety")
            elif dcf_base_up >= 10:
                dcf_verdict = ("UNDERVALUED", "green", f"DCF intrinsic {dcf_base_up:+.1f}% above current price — moderate margin of safety")
            elif dcf_base_up >= -10:
                dcf_verdict = ("FAIRLY VALUED", "yellow", f"DCF intrinsic within ±10% of current price — limited margin of safety")
            elif dcf_base_up >= -25:
                dcf_verdict = ("OVERVALUED", "red", f"DCF intrinsic {dcf_base_up:.1f}% below current price — no margin of safety")
            else:
                dcf_verdict = ("SIGNIFICANTLY OVERVALUED", "red", f"DCF intrinsic {dcf_base_up:.1f}% below current price — avoid")
            mos_signals.append({
                "method":  "DCF (Base Case)",
                "iv":      round(dcf_base_iv, 2),
                "upside":  dcf_base_up,
                "verdict": dcf_verdict[0],
                "color":   dcf_verdict[1],
                "note":    dcf_verdict[2],
            })

        # 2. Graham Number: √(22.5 × EPS × BVPS)
        eps     = info.get("trailingEps")
        bvps    = info.get("bookValue")
        graham  = None
        if eps and bvps and float(eps) > 0 and float(bvps) > 0:
            graham = round((22.5 * float(eps) * float(bvps)) ** 0.5, 2)
            g_up   = round((graham - float(cur_price)) / float(cur_price) * 100, 1)
            if g_up >= 20:
                gv = ("UNDERVALUED", "green", f"Graham Number {graham:.2f} — {g_up:+.1f}% above price")
            elif g_up >= -5:
                gv = ("FAIRLY VALUED", "yellow", f"Graham Number {graham:.2f} — {g_up:+.1f}% vs price")
            else:
                gv = ("OVERVALUED", "red", f"Graham Number {graham:.2f} — {g_up:.1f}% below price")
            mos_signals.append({
                "method":  "Graham Number",
                "iv":      graham,
                "upside":  g_up,
                "verdict": gv[0],
                "color":   gv[1],
                "note":    gv[2],
            })

        # 3. PEG Ratio (P/E ÷ earnings growth)
        pe_ratio      = info.get("trailingPE")
        earnings_grow = info.get("earningsGrowth") or info.get("revenueGrowth")
        peg           = None
        if pe_ratio and earnings_grow and float(earnings_grow) > 0:
            peg = round(float(pe_ratio) / (float(earnings_grow) * 100), 2)
            if peg < 0.5:
                pgv = ("VERY CHEAP", "green", f"PEG {peg} — deeply undervalued relative to growth")
            elif peg < 1.0:
                pgv = ("UNDERVALUED", "green", f"PEG {peg} — trading below growth rate (classic value signal)")
            elif peg < 1.5:
                pgv = ("FAIRLY VALUED", "yellow", f"PEG {peg} — fairly priced for its growth rate")
            elif peg < 2.5:
                pgv = ("OVERVALUED", "red", f"PEG {peg} — paying a premium over growth rate")
            else:
                pgv = ("EXPENSIVE", "red", f"PEG {peg} — significantly expensive vs. growth")
            mos_signals.append({
                "method":  "PEG Ratio",
                "iv":      None,
                "upside":  None,
                "peg":     peg,
                "verdict": pgv[0],
                "color":   pgv[1],
                "note":    pgv[2],
            })

        # 4. EV/EBITDA vs. sector median benchmarks
        SECTOR_EV_EBITDA = {
            "Technology": 25, "Healthcare": 18, "Communication Services": 15,
            "Consumer Discretionary": 14, "Financials": 12, "Industrials": 13,
            "Consumer Staples": 14, "Energy": 8, "Materials": 10,
            "Real Estate": 20, "Utilities": 12,
        }
        ev_ebitda = None
        if enterprise_value and ebitda_val and float(ebitda_val) > 0:
            ev_ebitda     = round(float(enterprise_value) / float(ebitda_val), 1)
            sector_median = SECTOR_EV_EBITDA.get(sector, 14)
            premium       = round((ev_ebitda - sector_median) / sector_median * 100, 1)
            if premium <= -20:
                evv = ("UNDERVALUED", "green", f"EV/EBITDA {ev_ebitda}x vs. sector median {sector_median}x — {abs(premium):.0f}% discount")
            elif premium <= 5:
                evv = ("IN LINE", "yellow", f"EV/EBITDA {ev_ebitda}x vs. sector median {sector_median}x — roughly in line")
            elif premium <= 30:
                evv = ("PREMIUM", "yellow", f"EV/EBITDA {ev_ebitda}x vs. sector median {sector_median}x — {premium:.0f}% premium")
            else:
                evv = ("EXPENSIVE", "red", f"EV/EBITDA {ev_ebitda}x vs. sector median {sector_median}x — {premium:.0f}% premium")
            mos_signals.append({
                "method":   "EV/EBITDA",
                "iv":       None,
                "upside":   None,
                "multiple": ev_ebitda,
                "sector_median": sector_median,
                "premium":  premium,
                "verdict":  evv[0],
                "color":    evv[1],
                "note":     evv[2],
            })

        # 5. Analyst consensus gap (current vs. mean target)
        target_mean   = info.get("targetMeanPrice")
        analyst_up    = None
        if target_mean and cur_price:
            analyst_up = round((float(target_mean) - float(cur_price)) / float(cur_price) * 100, 1)
            n_analysts  = info.get("numberOfAnalystOpinions", 0) or 0
            if analyst_up >= 20:
                av = ("STRONG BUY", "green", f"Wall Street target {analyst_up:+.1f}% above price ({n_analysts} analysts)")
            elif analyst_up >= 5:
                av = ("BUY", "green", f"Wall Street target {analyst_up:+.1f}% above price ({n_analysts} analysts)")
            elif analyst_up >= -5:
                av = ("HOLD", "yellow", f"Wall Street target in line with price ({n_analysts} analysts)")
            else:
                av = ("SELL", "red", f"Wall Street target {analyst_up:.1f}% below price ({n_analysts} analysts)")
            mos_signals.append({
                "method":  "Analyst Consensus",
                "iv":      round(float(target_mean), 2),
                "upside":  analyst_up,
                "verdict": av[0],
                "color":   av[1],
                "note":    av[2],
            })

        # Overall MoS score: weighted average of signal colors
        COLOR_SCORE = {"green": 2, "yellow": 1, "red": 0}
        if mos_signals:
            avg = sum(COLOR_SCORE.get(s["color"], 1) for s in mos_signals) / len(mos_signals)
            if avg >= 1.7:
                mos_overall = {"verdict": "STRONG VALUE", "color": "green",
                               "desc": "Multiple valuation methods agree the stock is undervalued with a meaningful margin of safety."}
            elif avg >= 1.2:
                mos_overall = {"verdict": "MODERATE VALUE", "color": "green",
                               "desc": "More signals point to undervaluation than overvaluation — some margin of safety present."}
            elif avg >= 0.8:
                mos_overall = {"verdict": "FAIRLY VALUED", "color": "yellow",
                               "desc": "Valuation signals are mixed — limited margin of safety at current price."}
            elif avg >= 0.4:
                mos_overall = {"verdict": "OVERVALUED", "color": "red",
                               "desc": "Most signals indicate the stock trades above intrinsic value — margin of safety is negative."}
            else:
                mos_overall = {"verdict": "SIGNIFICANTLY OVERVALUED", "color": "red",
                               "desc": "All valuation methods suggest the stock is materially overpriced."}
        else:
            mos_overall = {"verdict": "INSUFFICIENT DATA", "color": "yellow",
                           "desc": "Not enough data to calculate a meaningful margin of safety score."}

        result = {
            "ticker":        ticker,
            "currency":      currency,
            "currentPrice":  round(float(cur_price), 2),
            "fcfBase":       round(float(fcf) / 1e9, 3),   # in billions
            "sharesOut":     round(float(shares_out) / 1e9, 3),
            "netDebt":       round((total_debt - cash) / 1e9, 2),
            "wacc": {
                "total":        round(wacc * 100, 2),
                "costEquity":   round(cost_equity * 100, 2),
                "costDebt":     round(cost_debt * 100, 2),
                "costDebtPre":  round(cost_debt_pre * 100, 2),
                "equityWeight": round(equity_weight * 100, 1),
                "debtWeight":   round(debt_weight * 100, 1),
                "beta":         round(float(beta), 2),
                "rf":           round(RF * 100, 2),
                "erp":          round(ERP * 100, 2),
                "taxRate":      round(float(tax_rate) * 100, 1),
            },
            "scenarios":     scenario_results,
            "sensitivity": {
                "matrix":     matrix,
                "g_labels":   [f"{round(g*100,1)}%" for g in g_range],
                "wacc_labels":[f"{round(w*100,2)}%" for w in wacc_range],
                "base_g_idx": 3,
                "base_w_idx": 3,
            },
            "fcfTable":      fcf_table,
            "baseGrowth":    round(base_g * 100, 1),
            "sector":        sector,
            "mos":           mos_signals,
            "mosOverall":    mos_overall,
            "graham":        graham,
            "peg":           peg,
            "evEbitda":      ev_ebitda,
            "analystUpside": analyst_up,
        }
        cache_set(cache_key, result)
        return safe_jsonify(result)

    except Exception as e:
        logger.error(f"[api_dcf] {ticker}: {e}")
        return safe_jsonify({"error": str(e)}), 500


@app.route("/api/ai_analyse", methods=["POST"])
def api_ai_analyse():
    data       = request.json
    ticker     = data.get("ticker","").strip().upper()
    periode    = data.get("periode","1y")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if not ticker:
        return safe_jsonify({"error": "No ticker"}), 400
    if not gemini_key:
        return safe_jsonify({"error": "No API key configured"}), 400

    try:
        # Re-use cached data from /api/analyse to avoid duplicate downloads
        cache_key = f"analyse:{ticker}:{periode}"
        cached = cache_get(cache_key)

        if cached:
            fundamental = cached.get("fundamental", {})
            sig         = cached.get("signaler", {})
            ri          = cached.get("risiko", {})
            # For AI we need the raw yf.Ticker for insider/earnings/options context
            aksje = yf.Ticker(ticker)
            info_full = aksje.info
        else:
            # Cache miss — fetch fresh (runs in parallel with benchmark)
            aksje = yf.Ticker(ticker)
            def fetch_stock():
                df = yf.download(ticker, period=periode, progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                return df
            def fetch_bm():
                try:
                    bm = yf.download(BENCHMARK, period=periode, progress=False, auto_adjust=True)
                    if isinstance(bm.columns, pd.MultiIndex): bm.columns = bm.columns.get_level_values(0)
                    return bm
                except Exception as e:
                    logger.warning(f"[ai_analyse] benchmark fetch failed: {e}")
                    return None
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_df = ex.submit(fetch_stock)
                f_bm = ex.submit(fetch_bm)
                df   = f_df.result()
                bm   = f_bm.result()
            if df.empty:
                return safe_jsonify({"error": f"No data for {ticker}"}), 404
            info_full = aksje.info
            df  = beregn_tekniske(df)
            sig = hent_signaler(df)
            bm_df = bm if (bm is not None and not bm.empty) else df.copy()
            ri  = beregn_risiko(df, bm_df)
            fundamental = {
                "pe":           safe(info_full.get("trailingPE")),
                "forward_pe":   safe(info_full.get("forwardPE")),
                "mktcap":       stor_tall(info_full.get("marketCap")),
                "roe":          safe(info_full.get("returnOnEquity"), pst=True),
                "driftsmargin": safe(info_full.get("operatingMargins"), pst=True),
                "konsensus":    info_full.get("recommendationKey","N/A").upper(),
                "navn":         info_full.get("longName", ticker),
            }

        info = info_full

        analyst_ctx = ""
        try:
            target_mean = info.get("targetMeanPrice")
            target_high = info.get("targetHighPrice")
            target_low  = info.get("targetLowPrice")
            cur_price   = info.get("currentPrice") or info.get("regularMarketPrice")
            upside      = round((target_mean - cur_price) / cur_price * 100, 1) if target_mean and cur_price else None
            n_analysts  = info.get("numberOfAnalystOpinions", 0)
            strong_buy  = info.get("numberOfStrongBuyOpinions", 0) or info.get("strongBuy", 0)
            buy         = info.get("numberOfBuyOpinions", 0) or info.get("buy", 0)
            hold        = info.get("numberOfHoldOpinions", 0) or info.get("hold", 0)
            sell        = info.get("numberOfSellOpinions", 0) or info.get("sell", 0)
            strong_sell = info.get("numberOfStrongSellOpinions", 0) or info.get("strongSell", 0)
            analyst_ctx = f"Analyst consensus: {n_analysts} analysts — StrongBuy={strong_buy}, Buy={buy}, Hold={hold}, Sell={sell}, StrongSell={strong_sell}. Price targets: Low={target_low}, Mean={target_mean}, High={target_high}. Upside to mean target: {upside}%."
        except Exception as e:
            logger.warning(f"[ai_analyse] analyst context: {e}")

        insider_ctx = ""
        try:
            ins = aksje.insider_transactions
            if ins is not None and not ins.empty:
                ins = ins.head(10)
                buys  = sum(1 for _, r in ins.iterrows() if "buy" in str(r.get("Transaction","")).lower() or "purchase" in str(r.get("Transaction","")).lower())
                sells = len(ins) - buys
                recent = [f"{str(r.get('Insider',''))[:20]} ({str(r.get('Position',''))[:15]}): {str(r.get('Transaction',''))}"
                          for _, r in ins.head(5).iterrows()]
                insider_ctx = f"Insider activity (last 10): {buys} buys, {sells} sells. Recent: {'; '.join(recent)}."
        except Exception as e:
            logger.warning(f"[ai_analyse] insider context: {e}")

        earnings_ctx = ""
        try:
            cal = aksje.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        next_ed = str(ed[0])[:10] if hasattr(ed, '__len__') else str(ed)[:10]
                        earnings_ctx = f"Next earnings: {next_ed}."
            ei = aksje.earnings_history
            if ei is not None and not ei.empty:
                beats = misses = 0
                for _, r in ei.head(4).iterrows():
                    actual = r.get("epsActual") or r.get("Reported EPS")
                    est    = r.get("epsEstimate") or r.get("EPS Estimate")
                    if actual is not None and est is not None and actual == actual and est == est:
                        if float(actual) >= float(est): beats += 1
                        else: misses += 1
                earnings_ctx += f" Last 4 quarters: {beats} EPS beats, {misses} misses."
        except Exception as e:
            logger.warning(f"[ai_analyse] earnings context: {e}")

        short_ctx = ""
        try:
            short_pct    = info.get("shortPercentOfFloat")
            short_ratio  = info.get("shortRatio")
            shares_short = info.get("sharesShort")
            shares_short_prev = info.get("sharesShortPriorMonth")
            if short_pct is not None:
                short_pct_str = f"{round(float(short_pct)*100,1)}%"
                change_str = ""
                if shares_short and shares_short_prev and shares_short_prev > 0:
                    chg = (shares_short - shares_short_prev) / shares_short_prev * 100
                    change_str = f", changed {'+' if chg>=0 else ''}{round(chg,1)}% vs prior month"
                short_ctx = f"Short interest: {short_pct_str} of float shorted, days-to-cover={short_ratio}{change_str}."
        except Exception as e:
            logger.warning(f"[ai_analyse] short context: {e}")

        options_ctx = ""
        try:
            exps = aksje.options
            if exps:
                all_call_vol, all_put_vol, all_call_oi, all_put_oi = 0, 0, 0, 0
                for exp in exps[:4]:
                    try:
                        chain = aksje.option_chain(exp)
                        all_call_vol += chain.calls['volume'].fillna(0).sum()
                        all_put_vol  += chain.puts['volume'].fillna(0).sum()
                        all_call_oi  += chain.calls['openInterest'].fillna(0).sum()
                        all_put_oi   += chain.puts['openInterest'].fillna(0).sum()
                    except Exception as e:
                        logger.warning(f"[ai_analyse] options chain {exp}: {e}")
                pc_vol = round(all_put_vol / max(all_call_vol, 1), 2)
                pc_oi  = round(all_put_oi  / max(all_call_oi,  1), 2)
                atm_iv_str = ""
                try:
                    chain0    = aksje.option_chain(exps[0])
                    cur_price = info.get("currentPrice") or info.get("regularMarketPrice", 100)
                    calls     = chain0.calls.dropna(subset=['impliedVolatility'])
                    atm       = calls.iloc[(calls['strike'] - cur_price).abs().argsort()[:1]]
                    atm_iv    = round(float(atm['impliedVolatility'].values[0]) * 100, 1)
                    atm_iv_str = f", ATM implied volatility={atm_iv}%"
                except Exception as e:
                    logger.warning(f"[ai_analyse] ATM IV: {e}")
                sentiment   = "bearish" if pc_vol > 1.2 else "bullish" if pc_vol < 0.7 else "neutral"
                options_ctx = (f"Options flow (next 4 expirations): Put/Call volume ratio={pc_vol} ({sentiment} sentiment), "
                               f"Put/Call OI ratio={pc_oi}{atm_iv_str}.")
        except Exception as e:
            logger.warning(f"[ai_analyse] options context: {e}")

        prompt = f"""
You are a senior equity analyst at a professional investment firm. Provide a rigorous investment assessment for {fundamental['navn']} ({ticker}).

FUNDAMENTALS: P/E={fundamental['pe']}, Forward P/E={fundamental['forward_pe']}, Market Cap={fundamental['mktcap']}, ROE={fundamental['roe']}, Operating Margin={fundamental['driftsmargin']}, Analyst Consensus={fundamental['konsensus']}

TECHNICAL SIGNALS: RSI={sig['rsi']}, MACD={'Bullish crossover' if sig['macd_bull'] else 'Bearish crossover'}, Price vs SMA50={'Above — bullish' if sig['over_sma50'] else 'Below — bearish'}, Price vs SMA200={'Above — uptrend' if sig['over_sma200'] else 'Below — downtrend'}

RISK METRICS: Sharpe={ri.get('sharpe','N/A')}, Max Drawdown={ri.get('max_dd','N/A')}%, Beta={ri.get('beta','N/A')}, Sortino={ri.get('sortino','N/A')}

{f'WALL STREET: {analyst_ctx}' if analyst_ctx else ''}
{f'INSIDER ACTIVITY: {insider_ctx}' if insider_ctx else ''}
{f'EARNINGS TRACK RECORD: {earnings_ctx}' if earnings_ctx else ''}
{f'SHORT INTEREST: {short_ctx}' if short_ctx else ''}
{f'OPTIONS MARKET POSITIONING: {options_ctx}' if options_ctx else ''}

Structure your response exactly as follows:
1. **Overall Assessment** — 2-3 sentences synthesizing the complete picture across fundamentals, technicals, and sentiment
2. **Key Strengths** — 3 specific bullet points with data to support each
3. **Key Risks** — 3 specific bullet points, be candid about downside scenarios
4. **Smart Money Signals** — 1-2 sentences on what insiders, analysts, short sellers, and options traders are collectively signaling
5. **Entry Timing** — 1-2 sentences on whether technicals support buying now or waiting
6. **Verdict: BUY / HOLD / SELL** — one decisive sentence with the primary reason

Max 320 words. Be specific, data-driven, and direct. Avoid generic statements.
"""
        ai_tekst = spør_ai(prompt, gemini_key, 1500)
        return safe_jsonify({"ai": ai_tekst})

    except Exception as e:
        logger.error(f"[api_ai_analyse] {ticker}: {e}")
        return safe_jsonify({"error": str(e)}), 500

@app.route("/api/short_interest", methods=["POST"])
def api_short_interest():
    ticker = request.json.get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker"}), 400
    try:
        aksje = yf.Ticker(ticker)
        info  = aksje.info

        short_pct         = info.get("shortPercentOfFloat")
        short_ratio       = info.get("shortRatio")
        shares_short      = info.get("sharesShort")
        shares_short_prev = info.get("sharesShortPriorMonth")
        date_short        = info.get("dateShortInterest")
        float_shares      = info.get("floatShares")
        shares_out        = info.get("sharesOutstanding")

        # Month-over-month change
        mom_change = None
        if shares_short and shares_short_prev and shares_short_prev > 0:
            mom_change = round((shares_short - shares_short_prev) / shares_short_prev * 100, 1)

        # Interpret short level
        if short_pct is not None:
            pct = float(short_pct) * 100
            if pct > 20:   sentiment = ("HIGH", "var(--red)")
            elif pct > 10: sentiment = ("ELEVATED", "var(--yellow)")
            elif pct > 5:  sentiment = ("MODERATE", "var(--text3)")
            else:          sentiment = ("LOW", "var(--green)")
        else:
            sentiment = ("N/A", "var(--text2)")

        return safe_jsonify({
            "shortPct":        round(float(short_pct)*100, 2) if short_pct else None,
            "shortRatio":      round(float(short_ratio), 1) if short_ratio else None,
            "sharesShort":     stor_tall(shares_short) if shares_short else "N/A",
            "sharesShortPrev": stor_tall(shares_short_prev) if shares_short_prev else "N/A",
            "momChange":       mom_change,
            "floatShares":     stor_tall(float_shares) if float_shares else "N/A",
            "sharesOut":       stor_tall(shares_out) if shares_out else "N/A",
            "dateShort":       str(pd.Timestamp(date_short, unit='s'))[:10] if date_short else "N/A",
            "sentiment":       sentiment[0],
            "sentimentColor":  sentiment[1],
        })
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500


@app.route("/api/options_flow", methods=["POST"])
def api_options_flow():
    ticker = request.json.get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker"}), 400
    try:
        aksje     = yf.Ticker(ticker)
        info      = aksje.info
        exps      = aksje.options
        cur_price = info.get("currentPrice") or info.get("regularMarketPrice")

        if not exps:
            return safe_jsonify({"error": "No options data available for this ticker"}), 404

        # Aggregate across first 4 expirations for flow summary
        total_call_vol, total_put_vol = 0, 0
        total_call_oi,  total_put_oi  = 0, 0
        expirations_data = []

        for exp in exps[:3]:
            try:
                chain = aksje.option_chain(exp)
                cv = chain.calls['volume'].fillna(0).sum()
                pv = chain.puts['volume'].fillna(0).sum()
                ci = chain.calls['openInterest'].fillna(0).sum()
                pi = chain.puts['openInterest'].fillna(0).sum()
                total_call_vol += cv
                total_put_vol  += pv
                total_call_oi  += ci
                total_put_oi   += pi
                expirations_data.append({
                    "exp":      exp,
                    "callVol":  int(cv),
                    "putVol":   int(pv),
                    "callOI":   int(ci),
                    "putOI":    int(pi),
                    "pcVol":    round(pv / max(cv, 1), 2),
                })
            except: pass

        pc_vol = round(total_put_vol / max(total_call_vol, 1), 2)
        pc_oi  = round(total_put_oi  / max(total_call_oi,  1), 2)

        # Sentiment interpretation
        if pc_vol > 1.5:   flow_sentiment = ("VERY BEARISH", "var(--red)")
        elif pc_vol > 1.1: flow_sentiment = ("BEARISH",      "var(--red)")
        elif pc_vol < 0.5: flow_sentiment = ("VERY BULLISH", "var(--green)")
        elif pc_vol < 0.8: flow_sentiment = ("BULLISH",      "var(--green)")
        else:              flow_sentiment = ("NEUTRAL",       "var(--yellow)")

        # ATM implied volatility
        atm_iv = None
        try:
            chain0 = aksje.option_chain(exps[0])
            calls  = chain0.calls.dropna(subset=['impliedVolatility'])
            if cur_price and not calls.empty:
                atm  = calls.iloc[(calls['strike'] - cur_price).abs().argsort()[:1]]
                atm_iv = round(float(atm['impliedVolatility'].values[0]) * 100, 1)
        except: pass

        # Top 5 calls and puts by open interest (all strikes, nearest expiry)
        top_calls, top_puts = [], []
        try:
            chain0 = aksje.option_chain(exps[0])
            for _, r in chain0.calls.sort_values('openInterest', ascending=False).head(5).iterrows():
                top_calls.append({
                    "strike": round(float(r['strike']), 2),
                    "oi":     int(r['openInterest']) if r['openInterest'] == r['openInterest'] else 0,
                    "vol":    int(r['volume']) if r['volume'] == r['volume'] else 0,
                    "iv":     round(float(r['impliedVolatility'])*100, 1) if r['impliedVolatility'] == r['impliedVolatility'] else None,
                })
            for _, r in chain0.puts.sort_values('openInterest', ascending=False).head(5).iterrows():
                top_puts.append({
                    "strike": round(float(r['strike']), 2),
                    "oi":     int(r['openInterest']) if r['openInterest'] == r['openInterest'] else 0,
                    "vol":    int(r['volume']) if r['volume'] == r['volume'] else 0,
                    "iv":     round(float(r['impliedVolatility'])*100, 1) if r['impliedVolatility'] == r['impliedVolatility'] else None,
                })
        except: pass

        return safe_jsonify({
            "currentPrice":    round(float(cur_price), 2) if cur_price else None,
            "pcVolumeRatio":   pc_vol,
            "pcOIRatio":       pc_oi,
            "totalCallVol":    int(total_call_vol),
            "totalPutVol":     int(total_put_vol),
            "totalCallOI":     int(total_call_oi),
            "totalPutOI":      int(total_put_oi),
            "atmIV":           atm_iv,
            "sentiment":       flow_sentiment[0],
            "sentimentColor":  flow_sentiment[1],
            "expirations":     expirations_data,
            "topCalls":        top_calls,
            "topPuts":         top_puts,
            "nearestExp":      exps[0],
        })
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500


@app.route("/api/markeds_oversikt", methods=["POST"])
def api_markeds_oversikt():
    data     = request.json
    makro    = data.get("makro", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        return safe_jsonify({"error": "No API key configured"}), 400
    try:
        from datetime import datetime
        today = datetime.now().strftime("%A, %B %d, %Y")
        prompt = f"""
You are a senior macro strategist providing a daily market briefing for professional investors at Trade Wind Partners.

Today is {today}.

CURRENT MARKET DATA:
{makro}

Write a concise professional market briefing covering:
1. **Market Pulse** — overall risk-on/risk-off tone and what's driving it today
2. **Key Movers** — which instruments stand out and why (focus on unusual moves)
3. **Fixed Income & FX** — what yields and currency moves signal about macro expectations
4. **Upcoming Events** — key macro events to watch this week (Fed meetings, CPI, NFP, ECB, earnings seasons — use your knowledge of typical calendar)
5. **Trade Wind Outlook** — one actionable takeaway for equity investors today

Keep it professional, data-driven, and concise. Max 300 words.
"""
        ai_tekst = spør_ai(prompt, gemini_key, 1200)
        return safe_jsonify({"ai": ai_tekst})
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500


@app.route("/api/makro", methods=["GET"])
def api_makro():
    cached = cache_get("makro")
    if cached:
        return safe_jsonify(cached)
    def fetch_one(item):
        navn, sym = item
        try:
            t = yf.Ticker(sym)
            fi = t.fast_info
            siste   = float(fi.last_price)
            forrige = float(fi.previous_close) if fi.previous_close else siste
            if not forrige or forrige == 0: forrige = siste
            endring = (siste - forrige) / forrige * 100
            # Format value nicely
            if siste >= 1000:
                verdi = f"{siste:,.2f}"
            elif siste >= 1:
                verdi = f"{siste:.2f}"
            else:
                verdi = f"{siste:.4f}"
            return {"navn": navn, "verdi": verdi, "endring": round(endring, 2)}
        except Exception as e:
            return {"navn": navn, "verdi": "N/A", "endring": 0}
    with ThreadPoolExecutor(max_workers=8) as ex:
        rader = list(ex.map(fetch_one, MAKRO_TICKERS.items()))
    cache_set("makro", rader)
    return safe_jsonify(rader)

@app.route("/api/earnings", methods=["POST"])
def api_earnings():
    ticker = request.json.get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker provided"}), 400
    try:
        aksje = yf.Ticker(ticker)

        # Next earnings date
        next_date  = None
        next_time  = None
        next_est   = None
        try:
            cal = aksje.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        next_date = str(ed[0])[:10] if hasattr(ed, '__len__') else str(ed)[:10]
                elif hasattr(cal, 'columns'):
                    if "Earnings Date" in cal.columns:
                        next_date = str(cal["Earnings Date"].iloc[0])[:10]
                    elif "Earnings Date" in cal.index:
                        next_date = str(cal.loc["Earnings Date"].iloc[0])[:10]
        except: pass

        # Earnings history
        history = []
        try:
            ei = aksje.earnings_history
            if ei is not None and not ei.empty:
                ei = ei.sort_index(ascending=False).head(8)
                for idx, row in ei.iterrows():
                    actual   = round(float(row.get("epsActual",   row.get("Reported EPS", None))), 4) if row.get("epsActual",   row.get("Reported EPS", None)) is not None else None
                    estimate = round(float(row.get("epsEstimate", row.get("EPS Estimate", None))), 4) if row.get("epsEstimate", row.get("EPS Estimate", None)) is not None else None
                    try:
                        actual   = None if (actual   is not None and (actual   != actual))   else actual
                        estimate = None if (estimate is not None and (estimate != estimate)) else estimate
                    except: pass
                    history.append({
                        "date":     str(idx)[:10],
                        "actual":   actual,
                        "estimate": estimate,
                    })
        except:
            try:
                ei = aksje.earnings_dates
                if ei is not None and not ei.empty:
                    ei = ei.dropna(how="all").head(8)
                    for idx, row in ei.iterrows():
                        actual   = float(row["Reported EPS"])  if "Reported EPS"  in row and row["Reported EPS"]  == row["Reported EPS"]  else None
                        estimate = float(row["EPS Estimate"])  if "EPS Estimate"  in row and row["EPS Estimate"]  == row["EPS Estimate"]  else None
                        history.append({"date": str(idx)[:10], "actual": actual, "estimate": estimate})
            except: pass

        return safe_jsonify({
            "next_earnings": {"date": next_date, "time": next_time, "estimate": next_est} if next_date else None,
            "history": history
        })
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500

@app.route("/api/analyst", methods=["POST"])
def api_analyst():
    ticker = request.json.get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker provided"}), 400
    try:
        aksje = yf.Ticker(ticker)
        info  = aksje.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        target_mean   = info.get("targetMeanPrice")
        target_high   = info.get("targetHighPrice")
        target_low    = info.get("targetLowPrice")
        currency      = info.get("currency","")

        upside = None
        if current_price and target_mean:
            upside = round((target_mean - current_price) / current_price * 100, 1)

        # Ratings
        strong_buy  = info.get("numberOfAnalystOpinions")
        rec         = info.get("recommendationKey","")
        buy         = info.get("recommendationMean")

        # Detailed counts from analyst ratings
        try:
            ratings = aksje.analyst_price_targets
        except: ratings = None

        # Upgrades/downgrades — map Yahoo action codes to readable labels
        ACTION_MAP = {
            "main":      ("MAINTAIN",  "var(--text2)", 1),
            "reit":      ("REITERATE", "var(--text2)", 1),
            "init":      ("INITIATED", "var(--blue)",  3),
            "up":        ("UPGRADE",   "var(--green)", 4),
            "down":      ("DOWNGRADE", "var(--red)",   4),
            "upgrade":   ("UPGRADE",   "var(--green)", 4),
            "downgrade": ("DOWNGRADE", "var(--red)",   4),
            "resume":    ("RESUMED",   "var(--blue)",  3),
            "suspend":   ("SUSPENDED", "var(--yellow)",3),
            "coverage":  ("INITIATED", "var(--blue)",  3),
            "new":       ("INITIATED", "var(--blue)",  3),
        }

        upgrades = []
        try:
            ug = aksje.upgrades_downgrades
            if ug is not None and not ug.empty:
                ug = ug.sort_index(ascending=False).head(100)
                for idx, row in ug.iterrows():
                    raw_action = str(row.get("Action","")).lower().strip()
                    from_grade = str(row.get("From Grade","")).strip()
                    to_grade   = str(row.get("To Grade","")).strip()
                    if from_grade.lower() in ("nan","none",""): from_grade = ""
                    if to_grade.lower()   in ("nan","none",""): to_grade   = ""

                    # Infer upgrade/downgrade from grade change
                    if raw_action in ("main","reit") and from_grade and to_grade and from_grade.lower() != to_grade.lower():
                        grade_rank = {
                            "strong buy":5,"outperform":4,"overweight":4,"buy":4,
                            "market perform":3,"neutral":3,"equal-weight":3,"hold":3,"sector perform":3,
                            "underperform":2,"underweight":2,"sell":1,"strong sell":1
                        }
                        f = grade_rank.get(from_grade.lower(), 3)
                        t = grade_rank.get(to_grade.lower(), 3)
                        if t > f:   raw_action = "upgrade"
                        elif t < f: raw_action = "downgrade"

                    label, color, priority = ACTION_MAP.get(raw_action, (raw_action.upper(), "var(--text2)", 2))

                    if from_grade and to_grade and from_grade.lower() != to_grade.lower():
                        rating = f"{from_grade} → {to_grade}"
                    elif to_grade:
                        rating = to_grade
                    elif from_grade:
                        rating = from_grade
                    else:
                        rating = ""

                    # Skip only plain maintain/reiterate with absolutely no rating
                    if priority <= 1 and not rating:
                        continue

                    upgrades.append({
                        "date":   str(idx)[:10],
                        "firm":   str(row.get("Firm","")),
                        "action": label,
                        "color":  color,
                        "rating": rating,
                    })
                    if len(upgrades) >= 20:
                        break
        except: pass

        strong_buy_n = int(info.get("numberOfStrongBuyOpinions") or info.get("strongBuy") or 0)
        buy_n        = int(info.get("numberOfBuyOpinions")       or info.get("buy")       or 0)
        hold_n       = int(info.get("numberOfHoldOpinions")      or info.get("hold")      or 0)
        sell_n       = int(info.get("numberOfSellOpinions")      or info.get("sell")      or 0)
        strong_sell_n= int(info.get("numberOfStrongSellOpinions")or info.get("strongSell")or 0)

        # Fallback: try recommendations_summary if counts are all zero
        if strong_buy_n + buy_n + hold_n + sell_n + strong_sell_n == 0:
            try:
                rs = aksje.recommendations_summary
                if rs is not None and not rs.empty:
                    latest = rs.iloc[0]
                    strong_buy_n = int(latest.get("strongBuy",  0) or 0)
                    buy_n        = int(latest.get("buy",        0) or 0)
                    hold_n       = int(latest.get("hold",       0) or 0)
                    sell_n       = int(latest.get("sell",       0) or 0)
                    strong_sell_n= int(latest.get("strongSell", 0) or 0)
            except: pass

        return safe_jsonify({
            "currentPrice": round(current_price,2) if current_price else None,
            "targetMean":   round(target_mean,2)   if target_mean   else None,
            "targetHigh":   round(target_high,2)   if target_high   else None,
            "targetLow":    round(target_low,2)    if target_low    else None,
            "currency":     currency,
            "upside":       upside,
            "strongBuy":    strong_buy_n,
            "buy":          buy_n,
            "hold":         hold_n,
            "sell":         sell_n,
            "strongSell":   strong_sell_n,
            "upgrades":     upgrades,
        })
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500

@app.route("/api/insider", methods=["POST"])
def api_insider():
    ticker = request.json.get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker provided"}), 400
    try:
        aksje = yf.Ticker(ticker)
        transactions = []
        try:
            ins = aksje.insider_transactions
            if ins is not None and not ins.empty:
                ins = ins.head(25)
                for _, row in ins.iterrows():
                    shares = row.get("Shares") or row.get("shares")
                    value  = row.get("Value") or row.get("value") or row.get("Start Date")
                    try: shares = int(shares) if shares == shares else None
                    except: shares = None
                    try: value = int(value) if value == value else None
                    except: value = None
                    transactions.append({
                        "date":   str(row.get("Start Date", row.get("Date","")))[:10],
                        "name":   str(row.get("Insider",""))[:40],
                        "title":  str(row.get("Position", row.get("Title","")))[:30],
                        "type":   str(row.get("Transaction","")).strip(),
                        "shares": shares,
                        "value":  value,
                    })
        except: pass

        if not transactions:
            try:
                ins2 = aksje.insider_purchases
                if ins2 is not None and not ins2.empty:
                    for _, row in ins2.head(15).iterrows():
                        transactions.append({
                            "date":   str(row.get("Date",""))[:10],
                            "name":   str(row.get("Insider",""))[:40],
                            "title":  str(row.get("Position",""))[:30],
                            "type":   "Purchase",
                            "shares": int(row["Shares"]) if "Shares" in row else None,
                            "value":  int(row["Value"])  if "Value"  in row else None,
                        })
            except: pass

        return safe_jsonify({"transactions": transactions})
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500


@app.route("/api/sammenlign", methods=["POST"])
def api_sammenlign():
    data     = request.json
    tickers  = data.get("tickers", [])
    periode  = data.get("periode", "1y")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    resultat = []

    # Download benchmark once outside the loop
    bm_ret = None
    try:
        bm = yf.download(BENCHMARK, period="1y", progress=False, auto_adjust=True)
        if not bm.empty:
            if isinstance(bm.columns, pd.MultiIndex): bm.columns = bm.columns.get_level_values(0)
            bm_ret = bm["Close"].pct_change().dropna()
    except: pass

    for t in tickers:
        try:
            aksje = yf.Ticker(t)
            info  = aksje.info
            # Always use 1y — covers all sub-periods, no extra downloads needed
            df = yf.download(t, period="1y", progress=False, auto_adjust=True)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            last   = float(df["Close"].iloc[-1])
            ret    = df["Close"].pct_change().dropna()
            avk    = (last / float(df["Close"].iloc[0]) - 1) * 100
            vol    = ret.std() * np.sqrt(252) * 100
            sharpe = (ret.mean() - RISIKOFRI_RENTE/252) / ret.std() * np.sqrt(252)
            dd     = ((df["Close"] - df["Close"].cummax()) / df["Close"].cummax()).min() * 100

            # Beta using pre-downloaded benchmark
            beta = None
            if bm_ret is not None:
                felles = ret.index.intersection(bm_ret.index)
                if len(felles) > 10:
                    kov  = np.cov(ret.loc[felles], bm_ret.loc[felles])
                    beta = round(kov[0,1]/kov[1,1], 2) if kov[1,1] != 0 else None

            # Period returns from existing data — no extra downloads
            def sub_pct(n):
                try: return round((last / float(df["Close"].iloc[-n]) - 1) * 100, 2)
                except: return None

            avk_data = {"1M": sub_pct(21), "3M": sub_pct(63), "6M": sub_pct(126), "1Y": round(avk, 2)}

            historikk = {
                "datoer": [str(d)[:10] for d in df.index[-60:]],
                "priser": [round(float(p), 2) for p in df["Close"].iloc[-60:]]
            }

            resultat.append({
                "ticker":       t,
                "navn":         info.get("longName", t),
                "sektor":       info.get("sector", "N/A"),
                "pris":         safe(info.get("currentPrice")),
                "mktcap":       stor_tall(info.get("marketCap")),
                "pe":           safe(info.get("trailingPE")),
                "forward_pe":   safe(info.get("forwardPE")),
                "utbytte":      safe(info.get("dividendYield"), pst=True),
                "roe":          safe(info.get("returnOnEquity"), pst=True),
                "bruttomargin": safe(info.get("grossMargins"), pst=True),
                "driftsmargin": safe(info.get("operatingMargins"), pst=True),
                "konsensus":    info.get("recommendationKey","N/A").upper(),
                "mål":          safe(info.get("targetMeanPrice")),
                "avk":          round(avk, 2),
                "avk_perioder": avk_data,
                "vol":          round(vol, 2),
                "sharpe":       round(sharpe, 3),
                "max_dd":       round(dd, 2),
                "beta":         beta,
                "historikk":    historikk,
            })
        except: pass

    # AI comparison
    ai_tekst = ""
    if gemini_key and len(resultat) >= 2:
        summary = "\n".join([
            f"{r['ticker']} ({r['navn']}): P/E={r['pe']}, Sharpe={r['sharpe']}, "
            f"Return={r['avk']}%, Volatility={r['vol']}%, MaxDD={r['max_dd']}%, "
            f"Sector={r['sektor']}, Consensus={r['konsensus']}"
            for r in resultat
        ])
        prompt = f"""
You are a senior equity analyst. A client wants to allocate capital and is comparing these stocks. Give a decisive comparative analysis.

STOCKS:
{summary}

Your analysis must cover:
1. **Best Risk-Adjusted Return** — which stock offers the best Sharpe ratio and why it matters
2. **Best Value** — which is most attractively priced on fundamentals (P/E, sector context)
3. **Best Momentum** — which has the strongest recent performance and technical setup
4. **Avoid** — which stock you would not buy right now and the specific reason
5. **Final Ranking** — rank all stocks from most to least attractive right now, with one sentence per stock

Be decisive. A client is making a real allocation decision. Max 300 words.
"""
        ai_tekst = spør_ai(prompt, gemini_key, 1200)

    return safe_jsonify({"aksjer": resultat, "ai": ai_tekst})

@app.route("/api/portefolje_analyse", methods=["POST"])
def api_portefolje_analyse():
    data       = request.json
    posisjoner = data.get("posisjoner", [])
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if not posisjoner:
        return safe_jsonify({"error": "No positions provided"}), 400

    resultat      = []
    total_verdi   = 0
    total_kostnad = 0

    for pos in posisjoner:
        try:
            t       = pos["ticker"].upper()
            antall  = float(pos["antall"])
            snitt   = float(pos["snittpris"])
            kostnad = antall * snitt

            aksje = yf.Ticker(t)
            info  = aksje.info
            df    = yf.download(t, period="1y", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            curr_price = float(df["Close"].iloc[-1]) if not df.empty else snitt
            curr_value = antall * curr_price
            gain       = curr_value - kostnad
            gain_pct   = (gain / kostnad) * 100

            ret    = df["Close"].pct_change().dropna()
            vol    = ret.std()*np.sqrt(252)*100
            sharpe = (ret.mean()-RISIKOFRI_RENTE/252)/ret.std()*np.sqrt(252)

            total_verdi   += curr_value
            total_kostnad += kostnad

            # Price history (90 days)
            historikk = {
                "datoer": [str(d)[:10] for d in df.index[-90:]],
                "priser": [round(float(p),2) for p in df["Close"].iloc[-90:]]
            }

            resultat.append({
                "ticker":    t,
                "navn":      info.get("longName", t),
                "sektor":    info.get("sector","N/A"),
                "antall":    antall,
                "snittpris": snitt,
                "nåpris":    round(curr_price, 2),
                "kostnad":   round(kostnad, 2),
                "nåverdi":   round(curr_value, 2),
                "gevinst":   round(gain, 2),
                "pst":       round(gain_pct, 2),
                "vol":       round(vol, 2),
                "sharpe":    round(sharpe, 3),
                "pe":        safe(info.get("trailingPE")),
                "konsensus": info.get("recommendationKey","N/A").upper(),
                "mål":       safe(info.get("targetMeanPrice")),
                "historikk": historikk,
            })
        except Exception as e:
            resultat.append({"ticker": pos.get("ticker","?"), "feil": str(e)})

    total_gain = total_verdi - total_kostnad
    total_pst  = (total_gain/total_kostnad*100) if total_kostnad else 0

    # Sector breakdown
    sektorer = {}
    for r in resultat:
        if "feil" not in r:
            s = r.get("sektor","Unknown")
            sektorer[s] = sektorer.get(s,0) + r["nåverdi"]

    # AI analysis
    ai_tekst = ""
    if gemini_key and resultat:
        pos_info = "\n".join([
            f"{r['ticker']} ({r.get('navn',r['ticker'])}): "
            f"Shares={r.get('antall')}, AvgPrice={r.get('snittpris')}, "
            f"CurrentPrice={r.get('nåpris')}, Gain={r.get('pst','?')}%, "
            f"Sector={r.get('sektor','N/A')}, Sharpe={r.get('sharpe','N/A')}, "
            f"Consensus={r.get('konsensus','N/A')}, P/E={r.get('pe','N/A')}"
            for r in resultat if "feil" not in r
        ])
        prompt = f"""
You are a portfolio manager at a professional investment firm. Analyze this client portfolio and provide actionable recommendations.

PORTFOLIO SUMMARY (total value: ${round(total_verdi,2):,.2f}, total return: {round(total_pst,2)}%):
{pos_info}

SECTOR ALLOCATION: {json.dumps({k: f"{round(v/total_verdi*100,1)}%" for k,v in sektorer.items()})}

Provide a professional portfolio review:
1. **Portfolio Health** — is this portfolio well-constructed? Comment on diversification, concentration risk, and overall quality
2. **Top Performer** — the strongest position and whether to increase exposure
3. **Weakest Link** — the position with the worst risk/reward and a specific recommendation (sell, reduce, or hold with reasoning)
4. **Sector & Concentration Risk** — any dangerous overweights or missing sectors
5. **Gaps** — what asset classes, sectors, or geographies are missing for a balanced portfolio
6. **Position-by-Position** — for each holding: INCREASE / HOLD / REDUCE / EXIT with one-line reasoning
7. **Risk Rating: LOW / MEDIUM / HIGH** — overall portfolio risk with explanation

Be direct and action-oriented. This client needs clear guidance. Max 380 words.
"""
        ai_tekst = spør_ai(prompt, gemini_key, 1500)

    return safe_jsonify({
        "posisjoner":     resultat,
        "total_verdi":    round(total_verdi, 2),
        "total_kostnad":  round(total_kostnad, 2),
        "total_gevinst":  round(total_gain, 2),
        "total_pst":      round(total_pst, 2),
        "sektorer":       sektorer,
        "ai":             ai_tekst,
    })

@app.route("/api/nyheter", methods=["POST"])
def api_nyheter():
    data     = request.json
    ticker   = data.get("ticker","").strip().upper()
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not ticker:
        return safe_jsonify({"error": "No ticker provided"}), 400
    try:
        aksje  = yf.Ticker(ticker)
        nyheter = aksje.news or []
        resultat = []
        for n in nyheter[:10]:
            try:
                # yfinance news structure
                content = n.get("content", {})
                tittel  = content.get("title") or n.get("title","No title")
                kilde   = content.get("provider", {}).get("displayName","") or n.get("publisher","")
                lenke   = content.get("canonicalUrl", {}).get("url","") or n.get("link","")
                ts      = content.get("pubDate") or ""
                # Parse dato
                if ts:
                    try:
                        dato = datetime.strptime(ts[:10], "%Y-%m-%d").strftime("%d %b %Y")
                    except:
                        dato = ts[:10]
                else:
                    dato = ""
                if tittel:
                    resultat.append({"tittel": tittel, "kilde": kilde, "lenke": lenke, "dato": dato})
            except:
                continue

        # AI sentiment analysis
        ai_sentiment = ""
        if gemini_key and resultat:
            headlines = "\n".join([f"- {n['tittel']}" for n in resultat])
            prompt = f"""
You are a financial news analyst. Assess the market sentiment for {ticker} based on these recent headlines.

HEADLINES:
{headlines}

Provide:
1. **Sentiment: POSITIVE / NEUTRAL / NEGATIVE** — one word verdict
2. **Key Story** — the single most market-moving headline and why it matters for the stock
3. **Price Impact** — what this news flow suggests for near-term price action (be specific)

Max 120 words. Focus on what actually moves the stock price.
"""
            ai_sentiment = spør_ai(prompt, gemini_key, 400)

        return safe_jsonify({"nyheter": resultat, "ai_sentiment": ai_sentiment})
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500

@app.route("/api/watchlist_kurs", methods=["POST"])
def api_watchlist_kurs():
    data    = request.json
    tickers = data.get("tickers", [])
    if not tickers:
        return safe_jsonify([])

    def fetch_one(t):
        try:
            cached = cache_get(f"wl:{t}")
            if cached: return cached

            aksje = yf.Ticker(t)
            info  = aksje.info
            df = yf.download(t, period="1y", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if df.empty:
                return {"ticker": t, "feil": "No data"}

            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last

            def pct(n):
                try:
                    p = float(df["Close"].iloc[-n])
                    return round((last - p) / p * 100, 2)
                except: return None

            try:
                year_start = df[df.index.year == df.index[-1].year].iloc[0]
                ytd = round((last - float(year_start["Close"])) / float(year_start["Close"]) * 100, 2)
            except: ytd = None

            entry = {
                "ticker":    t,
                "navn":      info.get("longName", t),
                "sektor":    info.get("sector", "N/A"),
                "pris":      round(last, 2),
                "endring":   round((last - prev) / prev * 100, 2),
                "uke":       pct(5),
                "maaned":    pct(21),
                "tre_mnd":   pct(63),
                "ytd":       ytd,
                "høy52":     round(float(df["Close"].max()), 2),
                "lav52":     round(float(df["Close"].min()), 2),
                "mktcap":    stor_tall(info.get("marketCap")),
                "pe":        safe(info.get("trailingPE")),
                "konsensus": info.get("recommendationKey", "N/A").upper(),
                "sparkline": [round(float(p), 2) for p in df["Close"].iloc[-30:]],
            }
            cache_set(f"wl:{t}", entry)
            return entry
        except Exception as e:
            return {"ticker": t, "feil": str(e)[:50]}

    # Fetch all tickers in parallel (max 6 concurrent)
    resultat = [None] * len(tickers)
    with ThreadPoolExecutor(max_workers=min(len(tickers), 6)) as ex:
        futures = {ex.submit(fetch_one, t): i for i, t in enumerate(tickers)}
        for future in as_completed(futures):
            resultat[futures[future]] = future.result()

    return safe_jsonify([r for r in resultat if r is not None])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
