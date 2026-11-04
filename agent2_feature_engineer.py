"""
agent2_feature_engineer.py
----------------------------
Ajan 2: Quant Analyst (Teknik Özellik Çıkarımı)

Sorumluluk: SQL'den ham OHLCV okumak, pandas_ta ile indikatörler + özel
destek/direnç ve Fibonacci mesafe hesapları üretmek, sonucu DB'ye geri yazmak.
"""

import logging
import pandas as pd
import pandas_ta as ta

from config import CONFIG
from database import DatabaseManager, MarketData

logger = logging.getLogger(__name__)


class FeatureEngineerAgent:
    """Ham fiyat verisini zenginleştirilmiş özniteliklere dönüştüren ajan."""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()

    # ------------------------------------------------------------------
    # Veri okuma
    # ------------------------------------------------------------------
    def load_from_db(self, symbol: str) -> pd.DataFrame:
        query = f"""
            SELECT [date], [open], high, low, [close], volume
            FROM MarketData
            WHERE symbol = '{symbol}'
            ORDER BY [date] ASC
        """
        df = pd.read_sql(query, self.db.engine)
        df.columns = [c.lower() for c in df.columns]
        return df

    def load_full_from_db(self, symbol: str) -> pd.DataFrame:
        """Feature'lar dahil TÜM sütunları okur (Agent 3 / Agent 4 bunu kullanır)."""
        query = f"""
            SELECT *
            FROM MarketData
            WHERE symbol = '{symbol}'
            ORDER BY [date] ASC
        """
        df = pd.read_sql(query, self.db.engine)
        df.columns = [c.lower() for c in df.columns]
        return df

    @staticmethod
    def _find_col(frame: pd.DataFrame, prefix: str) -> str:
        """pandas_ta çıktısındaki sürüme göre değişen sütun ismini prefix ile bulur."""
        matches = [c for c in frame.columns if c.startswith(prefix)]
        if not matches:
            raise KeyError(f"'{prefix}' ile başlayan sütun bulunamadı: {list(frame.columns)}")
        return matches[0]

    # ------------------------------------------------------------------
    # Standart indikatörler (pandas_ta)
    # ------------------------------------------------------------------
    def add_standard_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["rsi_14"] = ta.rsi(df["close"], length=14)

        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]

        # pandas_ta sürümüne göre sütun isimleri değişebiliyor (örn. BBU_20_2.0
        # veya BBU_20_2.0_2.0) -> sabit isim yerine prefix ile eşleştiriyoruz.
        bbands = ta.bbands(df["close"], length=20, std=2)
        df["bb_upper"] = bbands[self._find_col(bbands, "BBU_")]
        df["bb_middle"] = bbands[self._find_col(bbands, "BBM_")]
        df["bb_lower"] = bbands[self._find_col(bbands, "BBL_")]

        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
        df["stoch_k"] = stoch["STOCHk_14_3_3"]
        df["stoch_d"] = stoch["STOCHd_14_3_3"]

        return df

    # ------------------------------------------------------------------
    # Kurumsal seviye ek göstergeler: ATR, ADX, OBV, rolling VWAP
    # ------------------------------------------------------------------
    def add_institutional_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ATR (volatilite), ADX (trend gücü), OBV (hacim akışı) ve rolling VWAP
        (kurumsal maliyet referansı) ekler.

        Not - VWAP: pandas_ta'nın ta.vwap() fonksiyonu intraday barlar için
        günlük anchor/reset mantığıyla çalışır; günlük (daily) barlarda her
        satır kendi başına "reset" olacağından anlamsız sonuç üretir. Bunun
        yerine burada N günlük KAYAN PENCERE (rolling) VWAP elle hesaplanır
        -- günlük grafiklerde kurumsal referans için standart pratik budur.
        """
        df = df.copy()

        # --- ATR: Average True Range (volatilite) ---
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=CONFIG.atr_length)
        df["atr_14"] = atr_series

        # --- ADX: trend gücü (0-100). >25 güçlü trend, <20 yatay piyasa kabul edilir ---
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=CONFIG.adx_length)
        df["adx_14"] = adx_df[f"ADX_{CONFIG.adx_length}"]

        # --- OBV: On-Balance Volume (kümülatif hacim akışı, fiyat teyidi için) ---
        df["obv"] = ta.obv(df["close"], df["volume"])

        # --- Rolling VWAP: N günlük kayan pencerede hacim ağırlıklı ortalama fiyat ---
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        pv = typical_price * df["volume"]
        window = CONFIG.vwap_window
        rolling_pv = pv.rolling(window=window, min_periods=1).sum()
        rolling_vol = df["volume"].rolling(window=window, min_periods=1).sum()
        df["vwap"] = rolling_pv / rolling_vol.replace(0, pd.NA)

        return df

    # ------------------------------------------------------------------
    # Kritik hesap 1: Dinamik destek / direnç (son N günün zirve-dip)
    # ------------------------------------------------------------------
    def add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        window = CONFIG.support_resistance_window
        df = df.copy()
        # rolling window: her satır için "kendisi dahil geçmiş N gün"deki max/min
        df["resistance_50d"] = df["high"].rolling(window=window, min_periods=1).max()
        df["support_50d"] = df["low"].rolling(window=window, min_periods=1).min()
        return df

    # ------------------------------------------------------------------
    # Kritik hesap 2: Fibonacci seviyeleri ve Golden Zone (0.618) mesafesi
    # ------------------------------------------------------------------
    def add_fibonacci_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        support_50d (swing low) ve resistance_50d (swing high) arasındaki
        aralığı temel alarak Fibonacci retracement seviyelerini hesaplar.
        dist_to_fib_618: kapanış fiyatının 0.618 seviyesine olan YÜZDE uzaklığı.
        Pozitif -> fiyat seviyenin üstünde, Negatif -> fiyat seviyenin altında.
        """
        df = df.copy()
        golden = CONFIG.golden_zone_level

        swing_range = df["resistance_50d"] - df["support_50d"]
        # Düşüşten yükselişe (retracement) mantığı: high'dan low'a doğru fib seviyeleri
        fib_618_level = df["resistance_50d"] - swing_range * golden
        df["fib_618"] = fib_618_level

        # sıfıra bölme koruması
        safe_level = fib_618_level.replace(0, pd.NA)
        df["dist_to_fib_618"] = ((df["close"] - fib_618_level) / safe_level) * 100

        # İstersen diğer seviyeleri de (0.236, 0.382, 0.5, 0.786) burada
        # ayrı sütunlar olarak ekleyebilirsin; CONFIG.fib_levels üzerinden döngü kurulabilir.
        return df

    # ------------------------------------------------------------------
    # Genişletilebilirlik: makro ekonomik veri placeholder
    # ------------------------------------------------------------------
    def add_macro_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PLACEHOLDER — İleride ECB faiz oranları, enflasyon, DXY endeksi gibi
        makro verileri buraya entegre edebilirsin. Örnek akış:
            1) Harici bir API'den (örn. FRED, ECB SDW) makro seriyi çek
            2) df["date"] ile tarihsel olarak merge_asof et (en yakın tarihe eşle)
            3) Yeni sütun olarak ekle (örn. df["ecb_rate"] = ...)
        Şimdilik hiçbir şey yapmadan df'i olduğu gibi döndürüyor.
        """
        # TODO: makro veri entegrasyonu
        return df

    # ------------------------------------------------------------------
    # Sonucu DB'ye geri yazma (mevcut satırları UPDATE eder)
    # ------------------------------------------------------------------
    def save_features_to_db(self, df: pd.DataFrame, symbol: str):
        feature_cols = [
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "stoch_k", "stoch_d",
            "atr_14", "vwap", "adx_14", "obv",
            "support_50d", "resistance_50d", "fib_618", "dist_to_fib_618",
        ]
        session = self.db.Session()
        try:
            records = df.to_dict("records")
            for rec in records:
                row = session.query(MarketData).filter_by(
                    symbol=symbol, date=rec["date"]
                ).first()
                if row is None:
                    continue
                for col in feature_cols:
                    val = rec.get(col)
                    setattr(row, col, None if pd.isna(val) else float(val))
            session.commit()
            logger.info(f"{symbol}: feature'lar DB'ye yazıldı ({len(records)} satır).")
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Orkestrasyon: tek sembol için tüm pipeline
    # ------------------------------------------------------------------
    def run(self, symbol: str) -> pd.DataFrame:
        df = self.load_from_db(symbol)
        if df.empty:
            logger.warning(f"{symbol} için DB'de veri yok. Önce Agent 1 çalıştırılmalı.")
            return df

        df = self.add_standard_indicators(df)
        df = self.add_institutional_indicators(df)
        df = self.add_support_resistance(df)
        df = self.add_fibonacci_distance(df)
        df = self.add_macro_data(df)  # placeholder, şu an no-op

        self.save_features_to_db(df, symbol)
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = FeatureEngineerAgent()
    for sym in CONFIG.symbols:
        result = agent.run(sym)
        print(f"{sym}: {len(result)} satır, {result.shape[1]} sütun")