"""
agent5_fundamental_analyst.py
--------------------------------
Ajan 5: Fundamental Analyst (Temel Analiz Uzmanı)

Sorumluluk:
  1) yfinance üzerinden çeyreklik Bilanço, Gelir Tablosu, Nakit Akım Tablosu
     ve genel şirket bilgilerini (info) çekmek.
  2) Kritik finansal rasyoları hesaplamak: ROE, Net Kar Marjı, F/K, PD/DD,
     Cari Oran, Borç/Özkaynak + çeyreklik büyüme yüzdeleri.
  3) Bu rasyoları bir LLM'e (Anthropic Claude API) gönderip "Warren Buffett
     tarzı" 3 maddelik bir değer yatırımı analizi ürettirmek.

NOT (önemli): yfinance'in çeyreklik tablo satır isimleri sürümden sürüme
değişebiliyor (örn. 'Net Income' vs 'NetIncome' vs 'Net Income Common
Stockholders'). Bu yüzden satırlar SABİT isimle değil, ADAY isim listesiyle
ve büyük/küçük harf duyarsız KISMİ EŞLEŞME ile aranıyor (_find_row metodu).
"""

import logging
import os
from typing import Optional

import pandas as pd
import yfinance as yf

from config import CONFIG

logger = logging.getLogger(__name__)


class FundamentalAnalystAgent:
    """Şirketin finansal sağlığını rasyolarla ölçen ve LLM ile yorumlatan ajan."""

    def __init__(self, llm_config=None):
        self.llm_config = llm_config or CONFIG.llm
        self._llm_client = None  # tembel (lazy) başlatılır, gereksiz import'u önler

    # ------------------------------------------------------------------
    # Görev 1: Ham finansal tabloları çekmek
    # ------------------------------------------------------------------
    def fetch_fundamentals(self, symbol: str) -> dict:
        """
        BIST hisseleri için de otomatik '.IS' fallback'i uygular (Agent 1
        ile aynı mantık). Bilanço, Gelir Tablosu, Nakit Akım Tablosu ve
        genel bilgileri (info) döndürür.
        """
        candidates = [symbol]
        if "." not in symbol:
            candidates.append(f"{symbol}.IS")

        for candidate in candidates:
            try:
                ticker = yf.Ticker(candidate)
                info = ticker.info or {}

                # yfinance sürümüne göre attribute adı değişebiliyor:
                # eski sürümler: quarterly_financials / eski sürüm sonrası: quarterly_income_stmt
                balance_sheet = self._safe_get(ticker, ["quarterly_balance_sheet", "quarterly_balancesheet"])
                income_stmt = self._safe_get(ticker, ["quarterly_income_stmt", "quarterly_financials"])
                cashflow = self._safe_get(ticker, ["quarterly_cashflow", "quarterly_cash_flow"])

                if balance_sheet is not None and not balance_sheet.empty:
                    if candidate != symbol:
                        logger.info(f"'{symbol}' bulunamadı, '{candidate}' (BIST) olarak eşleşti (fundamentals).")
                    return {
                        "symbol": symbol, "resolved_symbol": candidate,
                        "info": info, "balance_sheet": balance_sheet,
                        "income_stmt": income_stmt, "cashflow": cashflow,
                    }
            except Exception as e:
                logger.warning(f"{candidate} için fundamentals çekilirken hata: {e}")

        logger.warning(f"'{symbol}' için finansal tablo verisi bulunamadı.")
        return {"symbol": symbol, "resolved_symbol": None, "info": {},
                "balance_sheet": pd.DataFrame(), "income_stmt": pd.DataFrame(),
                "cashflow": pd.DataFrame()}

    @staticmethod
    def _safe_get(ticker, attr_names: list):
        """Birden fazla olası attribute isminden ilk çalışanı dener."""
        for name in attr_names:
            try:
                val = getattr(ticker, name, None)
                if val is not None and not (hasattr(val, "empty") and val.empty):
                    return val
            except Exception:
                continue
        return None

    @staticmethod
    def _find_row(df: pd.DataFrame, candidates: list) -> Optional[pd.Series]:
        """
        DataFrame'in index'inde (satır isimlerinde) aday isim listesinden
        birine kısmi/case-insensitive eşleşen ilk satırı döndürür.
        Bulamazsa None döner (hesaplama tarafı bunu NaN olarak ele alır).
        """
        if df is None or df.empty:
            return None
        index_lower = {str(idx).lower(): idx for idx in df.index}
        for cand in candidates:
            cand_lower = cand.lower()
            # 1) tam eşleşme
            if cand_lower in index_lower:
                return df.loc[index_lower[cand_lower]]
            # 2) kısmi eşleşme
            for idx_lower, idx_original in index_lower.items():
                if cand_lower in idx_lower:
                    return df.loc[idx_original]
        return None

    # ------------------------------------------------------------------
    # Görev 2: Rasyo hesaplama
    # ------------------------------------------------------------------
    def calculate_ratios(self, fundamentals: dict) -> dict:
        info = fundamentals.get("info", {})
        bs = fundamentals.get("balance_sheet")
        inc = fundamentals.get("income_stmt")

        ratios = {}

        # --- Değerleme rasyoları: doğrudan yfinance 'info' içinden ---
        ratios["pe_ratio"] = info.get("trailingPE")
        ratios["pb_ratio"] = info.get("priceToBook")
        ratios["market_cap"] = info.get("marketCap")

        # --- Bilanço satırları ---
        total_equity = self._find_row(bs, ["Total Stockholder Equity", "Stockholders Equity", "Total Equity"])
        total_liab = self._find_row(bs, ["Total Liab", "Total Liabilities Net Minority Interest", "Total Liabilities"])
        current_assets = self._find_row(bs, ["Total Current Assets", "Current Assets"])
        current_liab = self._find_row(bs, ["Total Current Liabilities", "Current Liabilities"])

        # --- Gelir tablosu satırları ---
        net_income = self._find_row(inc, ["Net Income", "Net Income Common Stockholders"])
        total_revenue = self._find_row(inc, ["Total Revenue", "Operating Revenue"])

        def latest(series):
            return series.iloc[0] if series is not None and len(series) > 0 else None

        eq_latest = latest(total_equity)
        liab_latest = latest(total_liab)
        ca_latest = latest(current_assets)
        cl_latest = latest(current_liab)
        ni_latest = latest(net_income)
        rev_latest = latest(total_revenue)

        # --- Karlılık ---
        ratios["roe_pct"] = self._safe_div(ni_latest, eq_latest, pct=True)
        ratios["net_profit_margin_pct"] = self._safe_div(ni_latest, rev_latest, pct=True)

        # --- Borç / Likidite ---
        ratios["current_ratio"] = self._safe_div(ca_latest, cl_latest)
        ratios["debt_to_equity"] = self._safe_div(liab_latest, eq_latest)

        # --- Çeyreklik büyüme (son çeyrek vs bir önceki çeyrek) ---
        ratios["revenue_qoq_growth_pct"] = self._qoq_growth(total_revenue)
        ratios["net_income_qoq_growth_pct"] = self._qoq_growth(net_income)

        return ratios

    @staticmethod
    def _safe_div(numerator, denominator, pct: bool = False):
        if numerator is None or denominator is None:
            return None
        try:
            if denominator == 0:
                return None
            result = numerator / denominator
            return round(result * 100, 2) if pct else round(result, 2)
        except (TypeError, ZeroDivisionError):
            return None

    @staticmethod
    def _qoq_growth(series) -> Optional[float]:
        """Son çeyrek vs bir önceki çeyrek yüzde değişim (Quarter-over-Quarter)."""
        if series is None or len(series) < 2:
            return None
        current, previous = series.iloc[0], series.iloc[1]
        if previous in (0, None) or current is None:
            return None
        try:
            return round(((current - previous) / abs(previous)) * 100, 2)
        except (TypeError, ZeroDivisionError):
            return None

    # ------------------------------------------------------------------
    # Görev 3: LLM ile "Warren Buffett tarzı" metin analizi
    # ------------------------------------------------------------------
    def build_prompt(self, symbol: str, ratios: dict) -> str:
        def fmt(key, suffix=""):
            val = ratios.get(key)
            return f"{val}{suffix}" if val is not None else "veri yok"

        return f"""Sen Warren Buffett tarzı, temkinli ve rasyo odaklı bir değer yatırımı analistisin.

Aşağıda {symbol} hissesinin güncel çeyreklik finansal rasyoları verilmiştir:

DEĞERLEME:
- F/K (P/E): {fmt('pe_ratio')}
- PD/DD (P/B): {fmt('pb_ratio')}
- Piyasa Değeri: {fmt('market_cap')}

KARLILIK:
- ROE (Özsermaye Karlılığı): {fmt('roe_pct', '%')}
- Net Kar Marjı: {fmt('net_profit_margin_pct', '%')}

BORÇ / LİKİDİTE:
- Cari Oran: {fmt('current_ratio')}
- Borç/Özkaynak Oranı: {fmt('debt_to_equity')}

ÇEYREKLİK BÜYÜME (bir önceki çeyreğe göre):
- Gelir Büyümesi: {fmt('revenue_qoq_growth_pct', '%')}
- Net Kâr Büyümesi: {fmt('net_income_qoq_growth_pct', '%')}

Bu rasyolara bakarak hissenin ucuz mu pahalı mı olduğunu ve öne çıkan
finansal risklerini TAM OLARAK 3 maddede, Türkçe, kısa ve öz şekilde özetle.
Rasyo verisi eksikse (\"veri yok\" yazan alanlar) bunu açıkça belirt, tahmin
yürütme. Yatırım tavsiyesi olmadığını son cümlede hatırlat."""

    def _get_llm_client(self):
        """Anthropic client'ı tembel (lazy) başlatır -- API anahtarı yoksa hiç import etmez."""
        if self._llm_client is None:
            import anthropic  # sadece gerektiğinde import edilir
            self._llm_client = anthropic.Anthropic(api_key=self.llm_config.api_key)
        return self._llm_client

    def generate_llm_report(self, prompt: str, symbol: str) -> str:
        """
        LLM'den 3 maddelik analiz metni üretir. API anahtarı yoksa veya
        çağrı başarısız olursa, pipeline'ı ÇÖKERTMEDEN kural-tabanlı basit
        bir uyarı metni döndürür (fail-safe).
        """
        if not self.llm_config.enabled:
            return "LLM raporu devre dışı (config.py -> llm.enabled=False)."

        if not self.llm_config.api_key:
            logger.warning("ANTHROPIC_API_KEY tanımlı değil, LLM raporu atlanıyor.")
            return ("LLM raporu üretilemedi: ANTHROPIC_API_KEY ortam değişkeni "
                    "tanımlı değil. Windows'ta 'setx ANTHROPIC_API_KEY \"sk-ant-...\"' "
                    "ile ayarlayıp terminali yeniden açman gerekiyor.")

        try:
            client = self._get_llm_client()
            message = client.messages.create(
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [block.text for block in message.content if block.type == "text"]
            return "\n".join(text_blocks).strip()
        except Exception as e:
            logger.error(f"{symbol} için LLM çağrısı başarısız: {e}")
            return f"LLM raporu üretilemedi (hata: {e})."

    # ------------------------------------------------------------------
    # Orkestrasyon: tek sembol için tüm Ajan 5 akışı
    # ------------------------------------------------------------------
    def run(self, symbol: str) -> dict:
        fundamentals = self.fetch_fundamentals(symbol)
        ratios = self.calculate_ratios(fundamentals)
        prompt = self.build_prompt(symbol, ratios)
        llm_report = self.generate_llm_report(prompt, symbol)

        return {
            "symbol": symbol,
            "ratios": ratios,
            "llm_report": llm_report,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = FundamentalAnalystAgent()
    result = agent.run(CONFIG.symbols[0])
    print("Rasyolar:", result["ratios"])
    print("\nLLM Raporu:\n", result["llm_report"])