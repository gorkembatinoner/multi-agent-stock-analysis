"""
agent1_data_collector.py
-------------------------
Ajan 1: Data Collector & Database Manager (Veri Mühendisi)

Sorumluluk: yfinance'ten OHLCV çekmek, temizlemek, SQL Server'a UPSERT ile yazmak.
"""

import logging
import pandas as pd
import yfinance as yf

from config import CONFIG
from database import DatabaseManager

logger = logging.getLogger(__name__)


class DataCollectorAgent:
    """Ham piyasa verisini toplayan ve kalıcı hale getiren ajan."""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()
        self.db.create_tables()

    def fetch_symbol(self, symbol: str) -> pd.DataFrame:
        """
        Tek bir sembol için yfinance'ten OHLCV çeker.
        BIST hisseleri yfinance'te '.IS' uzantısı gerektirir (örn. SANEL.IS).
        Kullanıcı düz kod girerse (örn. 'SANEL'), önce olduğu gibi denenir,
        boş dönerse otomatik olarak '.IS' eklenip tekrar denenir.
        """
        candidates = [symbol]
        if "." not in symbol:
            candidates.append(f"{symbol}.IS")

        for candidate in candidates:
            logger.info(f"{candidate} için veri çekiliyor...")
            ticker = yf.Ticker(candidate)
            df = ticker.history(period=CONFIG.history_period, interval=CONFIG.history_interval)

            if not df.empty:
                if candidate != symbol:
                    logger.info(f"'{symbol}' bulunamadı, '{candidate}' (BIST) olarak eşleşti.")
                df = df.reset_index()
                df = df.rename(columns={
                    "Date": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                })
                return self._clean(df[["date", "open", "high", "low", "close", "volume"]])

        logger.warning(f"'{symbol}' için veri bulunamadı (BIST '.IS' varyasyonu da denendi).")
        return pd.DataFrame()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """NaN temizliği ve tip dönüşümleri."""
        df = df.dropna(subset=["open", "high", "low", "close"])
        df["volume"] = df["volume"].fillna(0)

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=numeric_cols)

        # timezone bilgisini kaldır (SQL Server DATETIME ile uyum için)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df.reset_index(drop=True)

    def run(self, symbols=None) -> dict:
        """
        Tüm sembolleri sırayla çeker, temizler ve DB'ye yazar.
        Dönüş: {symbol: DataFrame} -> orkestratörün / Agent 2'nin RAM üzerinden
        de erişebilmesi için (DB'ye ek olarak) bellekte de tutulur.
        """
        symbols = symbols or CONFIG.symbols
        results = {}
        for symbol in symbols:
            try:
                df = self.fetch_symbol(symbol)
                if df.empty:
                    continue
                self.db.upsert_dataframe(df, symbol)
                results[symbol] = df
            except Exception as e:
                logger.error(f"{symbol} işlenirken hata: {e}")
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = DataCollectorAgent()
    data = agent.run()
    for sym, df in data.items():
        print(f"{sym}: {len(df)} satır çekildi ve DB'ye yazıldı.")