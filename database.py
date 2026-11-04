"""
database.py
-----------
SQLAlchemy ORM modeli (MarketData tablosu) ve engine/session yönetimi.
Ajan 1 buraya yazar, Ajan 2 buradan okur -> ajanlar arası "ortak dil" burasıdır.
"""

import logging
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    UniqueConstraint, text
)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import CONFIG

logger = logging.getLogger(__name__)

Base = declarative_base()


class MarketData(Base):
    """
    Ham OHLCV + Ajan 2'nin ekleyeceği feature sütunlarını da barındıran ana tablo.
    (symbol, date) çifti benzersizdir -> UPSERT bu constraint üzerinden çalışır.
    """
    __tablename__ = "MarketData"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)

    # --- Ajan 2 tarafından doldurulacak feature sütunları (nullable) ---
    rsi_14 = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_hist = Column(Float, nullable=True)
    bb_upper = Column(Float, nullable=True)
    bb_middle = Column(Float, nullable=True)
    bb_lower = Column(Float, nullable=True)
    stoch_k = Column(Float, nullable=True)
    stoch_d = Column(Float, nullable=True)

    # --- Kurumsal seviye ek göstergeler ---
    atr_14 = Column(Float, nullable=True)       # Volatilite (Average True Range)
    vwap = Column(Float, nullable=True)         # Hacim ağırlıklı ortalama fiyat (rolling)
    adx_14 = Column(Float, nullable=True)       # Trend gücü (Average Directional Index)
    obv = Column(Float, nullable=True)          # On-Balance Volume

    support_50d = Column(Float, nullable=True)
    resistance_50d = Column(Float, nullable=True)
    fib_618 = Column(Float, nullable=True)
    dist_to_fib_618 = Column(Float, nullable=True)  # yüzde olarak uzaklık

    # --- Ajan 3 tarafından doldurulacak ---
    ml_label = Column(Integer, nullable=True)       # eğitim etiketi: -1, 0, 1
    ml_prediction = Column(Integer, nullable=True)   # model çıktısı

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_symbol_date"),
    )


class DatabaseManager:
    """Engine/Session yaşam döngüsünü yöneten yardımcı sınıf."""

    def __init__(self, db_config=None):
        self.db_config = db_config or CONFIG.db
        self.engine = create_engine(self.db_config.connection_string, fast_executemany=True)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        Base.metadata.create_all(self.engine)
        logger.info("MarketData tablosu hazır (yoksa oluşturuldu).")

    def upsert_dataframe(self, df, symbol: str):
        """
        SQL Server'da native UPSERT olmadığı için MERGE komutu kullanılır.
        Akış: 1) DataFrame'i geçici (staging) tabloya yaz
              2) MERGE ile ana tabloya birleştir (varsa UPDATE, yoksa INSERT)
              3) Staging tabloyu sil
        """
        if df.empty:
            logger.warning(f"{symbol} için upsert edilecek veri yok.")
            return

        staging_table = f"##staging_{symbol.replace('.', '_')}"
        df = df.copy()
        df["symbol"] = symbol

        with self.engine.begin() as conn:
            df.to_sql(staging_table.strip("#"), conn, if_exists="replace", index=False)

            merge_sql = text(f"""
                MERGE INTO MarketData AS target
                USING {staging_table.strip('#')} AS source
                ON target.symbol = source.symbol AND target.[date] = source.[date]
                WHEN MATCHED THEN
                    UPDATE SET
                        target.[open] = source.[open],
                        target.high = source.high,
                        target.low = source.low,
                        target.[close] = source.[close],
                        target.volume = source.volume,
                        target.updated_at = GETUTCDATE()
                WHEN NOT MATCHED THEN
                    INSERT (symbol, [date], [open], high, low, [close], volume, updated_at)
                    VALUES (source.symbol, source.[date], source.[open], source.high,
                            source.low, source.[close], source.volume, GETUTCDATE());
            """)
            conn.execute(merge_sql)
            conn.execute(text(f"DROP TABLE IF EXISTS {staging_table.strip('#')}"))

        logger.info(f"{symbol}: {len(df)} satır UPSERT edildi.")