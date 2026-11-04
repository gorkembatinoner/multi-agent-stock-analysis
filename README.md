# Multi-Agent Stock Analysis Pipeline

An end-to-end stock analysis pipeline built from five specialized "agents". It collects market data, engineers technical features, trains an XGBoost classifier, generates an interactive chart, and produces an LLM-written fundamental analysis, all orchestrated from a single command.

> **Disclaimer:** This is an educational / research project. Nothing it outputs is financial advice. See [Limitations](#limitations).

---

## Features

- **Data collection** with yfinance, stored in SQL Server via an idempotent `MERGE` (upsert) so re-running never creates duplicates
- **Automatic BIST support**: enter `SANEL` and the pipeline retries as `SANEL.IS` if the plain ticker is not found
- **Technical indicators**: RSI, MACD, Bollinger Bands, Stochastic, ATR, ADX, OBV, rolling VWAP, 50-day support/resistance, Fibonacci 0.618 distance
- **ML signal model**: XGBoost classifier (Buy / Neutral / Sell) with ATR-based dynamic labeling, class weighting, and chronological `TimeSeriesSplit` validation (no random shuffling, so no look-ahead bias)
- **TradingView-style interactive chart** (Plotly, 3 panels: price + signals, volume + OBV, RSI/ADX/MACD)
- **Rule-based technical assessment** that is independent of the ML model and explains its score
- **Fundamental analysis**: ROE, net margin, P/E, P/B, current ratio, debt/equity, QoQ growth, summarized by the Claude API in three bullet points
- **JSON output** designed for consumption by a mobile/web client (e.g. Flutter)
- **Fail-safe design**: if one symbol or the LLM step fails, the rest of the pipeline keeps running

## Architecture

```
                    +---------------------+
                    |   Orchestrator      |
                    +----------+----------+
                               |
   +---------------+   +-------v-------+   +---------------+
   | Agent 1       |-->| Agent 2       |-->| Agent 3       |
   | Data Collector|   | Feature Eng.  |   | ML Engineer   |
   | yfinance + DB |   | indicators    |   | XGBoost       |
   +---------------+   +---------------+   +-------+-------+
                                                   |
   +---------------+                       +-------v-------+
   | Agent 5       |---------------------->| Agent 4       |
   | Fundamental   |   ratios + LLM report | Visualizer    |
   | Analyst (LLM) |                       | chart + JSON  |
   +---------------+                       +---------------+
```

| Agent | File | Responsibility |
|-------|------|----------------|
| 1 | `agent1_data_collector.py` | Fetch OHLCV from yfinance, clean it, upsert into SQL Server |
| 2 | `agent2_feature_engineer.py` | Read from DB, compute indicators, write features back |
| 3 | `agent3_ml_engineer.py` | Label data, train XGBoost with time-series CV, predict |
| 4 | `agent4_visualizer.py` | Plotly chart, JSON signal, rule-based assessment |
| 5 | `agent5_fundamental_analyst.py` | Financial ratios + Claude-generated value-investing summary |
| - | `orchestrator.py` | Runs the agents in order and handles errors |
| - | `config.py` | Single source of configuration (symbols, DB, model, LLM) |
| - | `database.py` | SQLAlchemy model and `MERGE`-based upsert |

## Requirements

- **Python 3.12** (newer versions may lack prebuilt wheels for `pandas_ta` / `numba`)
- **SQL Server** (Express edition is fine) with an empty database named `AlgoTradingDB`
- **ODBC Driver 17 (or 18) for SQL Server**
- *(Optional)* An [Anthropic API key](https://console.anthropic.com/) for the fundamental analysis report

## Installation

```bash
git clone https://github.com/<gorkembatinoner>/<multi-agent-stock-analysis>.git
cd <your-repo>

python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

All settings live in `config.py`.

**Database.** The default is a local SQL Server Express instance with Windows Authentication:

```python
server = r".\SQLEXPRESS"
database = "AlgoTradingDB"
driver = "ODBC Driver 17 for SQL Server"
trusted_connection = True
```

Create the database once in SQL Server Management Studio (`CREATE DATABASE AlgoTradingDB;`). The tables are created automatically on first run. For SQL authentication, set `trusted_connection = False` and fill in `username` / `password`.

**Claude API key (optional).** The key is read from an environment variable and is never stored in code:

```powershell
# Windows (then reopen the terminal)
setx ANTHROPIC_API_KEY "sk-ant-..."
```

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."
```

Without a key the pipeline still runs; the fundamental report simply says it could not be generated. You can also set `llm.enabled = False` in `config.py`. The model name is set by `llm.model` in `config.py`; make sure it is a valid model name for your account.

**Key pipeline parameters** (`config.py`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `history_period` | `2y` | How much history to download |
| `label_horizon_days` | `5` | Look-ahead window used to create training labels |
| `label_atr_multiplier` | `1.0` | A move larger than N x ATR counts as Buy/Sell |
| `n_splits` | `5` | Number of `TimeSeriesSplit` folds |

## Usage

```bash
# Pass a ticker as an argument
python orchestrator.py AAPL

# Or run without arguments and enter the ticker when prompted
python orchestrator.py

# Borsa Istanbul stocks work too (".IS" is added automatically)
python orchestrator.py SANEL
```

Each agent can also be run on its own for debugging, for example `python agent5_fundamental_analyst.py`.

### What you get

1. **Interactive chart**: an HTML file saved to the `outputs/` folder
2. **Technical assessment** printed to the terminal, with a score and the reasons behind it
3. **JSON signal** printed at the end of the run:

```json
{
  "AAPL": {
    "symbol": "AAPL",
    "date": "2026-11-03",
    "price": { "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0 },
    "indicators": { "rsi_14": 0.0, "macd": 0.0, "stoch_k": 0.0, "atr_14": 0.0,
                    "adx_14": 0.0, "vwap": 0.0, "obv": 0 },
    "levels": { "support_50d": 0.0, "resistance_50d": 0.0,
                "fib_618": 0.0, "dist_to_fib_618_pct": 0.0 },
    "signal": { "code": 0, "label": "YATAY_NOTR" },
    "fundamental_analysis": { "symbol": "AAPL", "ratios": { }, "llm_report": "..." }
  }
}
```

Signal labels: `UCUZ_AL_FIRSATI` (buy opportunity), `PAHALI_SAT_FIRSATI` (sell opportunity), `YATAY_NOTR` (neutral). Values above are placeholders.

## Design decisions

- **Time-series-aware validation.** `TimeSeriesSplit` guarantees the training set always precedes the test set chronologically, which avoids look-ahead bias.
- **ATR-based labels instead of a fixed percentage.** A fixed threshold fires rarely on calm stocks and constantly on volatile ones. ATR scales the target to each stock's own volatility. Lowering the multiplier from 2.0 to 1.0 fixed a ~97% "neutral" label imbalance.
- **Class weighting.** Balanced sample weights keep the model from defaulting to the majority "neutral" class.
- **Idempotent writes.** A staging table plus SQL Server `MERGE` makes re-runs safe.
- **Resilient by design.** Per-symbol `try/except` in the orchestrator, folds with a single class are skipped, and the LLM step degrades gracefully when no API key is available.
- **Robust to library changes.** yfinance and pandas_ta row/column names differ across versions, so rows are matched against candidate names with case-insensitive partial matching.

## Limitations

- The final model is trained on the full dataset and then predicts on that same data, so displayed signals are **in-sample** and not a measure of real predictive power. Use the per-fold results for an honest estimate.
- There is **no backtest** yet (no transaction costs, slippage, or position sizing).
- The model is retrained on every run and is not persisted.
- The fundamental report comes from an LLM given computed ratios; treat it as a summary, not as analysis you can rely on.
- Data comes from yfinance (unofficial) and may be delayed, incomplete, or change format.

## Roadmap

- [ ] Backtesting module with commissions and slippage
- [ ] Walk-forward validation
- [ ] Model persistence and versioning
- [ ] Feature importance / SHAP explanations
- [ ] Macro data (rates, inflation, DXY); a placeholder exists in Agent 2
- [ ] REST API (FastAPI) for the mobile client
- [ ] Parameterized SQL queries and unit tests
- [ ] Scheduled runs (cron / Airflow)

## Tech stack

Python 3.12, pandas, numpy, pandas_ta, scikit-learn, XGBoost, SQLAlchemy, pyodbc, SQL Server, Plotly, yfinance, Anthropic API

## License

Choose a license (for example [MIT](https://choosealicense.com/licenses/mit/)) and add a `LICENSE` file.
