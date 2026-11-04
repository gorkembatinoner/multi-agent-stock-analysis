"""
config.py
---------
Tüm pipeline'ın tek konfigürasyon kaynağı. Ajanlar birbirini değil,
bu dosyayı referans alır. Böylece sembol listesi, DB bağlantısı veya
model parametreleri değiştiğinde tek yerden yönetilir.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DBConfig:
    # Windows Authentication ile yerel SQLEXPRESS bağlantısı (pyodbc + SQLAlchemy)
    # Not: ODBC Driver 17/18 for SQL Server kurulu olmalı.
    server: str = r".\SQLEXPRESS"
    database: str = "AlgoTradingDB"
    driver: str = "ODBC Driver 17 for SQL Server"
    trusted_connection: bool = True  # Windows Auth -> True. SQL Auth kullanacaksan False yap.
    username: str = ""
    password: str = ""

    @property
    def connection_string(self) -> str:
        driver_enc = self.driver.replace(" ", "+")
        if self.trusted_connection:
            return (
                f"mssql+pyodbc://@{self.server}/{self.database}"
                f"?driver={driver_enc}&trusted_connection=yes"
            )
        return (
            f"mssql+pyodbc://{self.username}:{self.password}@{self.server}/{self.database}"
            f"?driver={driver_enc}"
        )


@dataclass
class LLMConfig:
    """
    Ajan 5'in temel analiz raporunu ürettirdiği LLM ayarları.
    API anahtarı KOD İÇİNE YAZILMAZ -- ortam değişkeninden okunur.
    Windows'ta ayarlamak için (kalıcı, kullanıcı seviyesinde):
        setx ANTHROPIC_API_KEY "sk-ant-..."
    (setx sonrası terminali YENİDEN AÇMAN gerekir.)
    """
    provider: str = "anthropic"          # şimdilik desteklenen: "anthropic"
    model: str = "claude-sonnet-5"
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    max_tokens: int = 800
    enabled: bool = True   # False yaparsan Ajan 5 LLM çağırmaz, kural-tabanlı özet üretir


@dataclass
class PipelineConfig:
    # Ajan 1 - Veri çekimi
    symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
    ])
    history_period: str = "2y"       # yfinance period: 1y, 2y, 5y, max...
    history_interval: str = "1d"     # 1d, 1h, ...

    # Ajan 2 - Feature engineering
    support_resistance_window: int = 50   # son N günün zirve/dip taraması
    fib_levels: List[float] = field(default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786])
    golden_zone_level: float = 0.618
    atr_length: int = 14
    adx_length: int = 14
    vwap_window: int = 20             # rolling VWAP penceresi (gün)

    # Ajan 3 - ML etiketleme & model
    label_horizon_days: int = 5      # ileriye dönük kaç gün bakılacak
    label_atr_multiplier: float = 1.0  # ATR tabanlı dinamik hedef: N x ATR
                                        # (2.0 çok nadir tetikleniyordu -> %97 "Yatay"
                                        # etiketi üretiyordu; 1.0 ile daha dengeli:
                                        # ~%71 Yatay / %14 Al / %14 Sat)
    test_size_ratio: float = 0.2     # time-series split için son dilim oranı
    n_splits: int = 5                # TimeSeriesSplit fold sayısı

    db: DBConfig = field(default_factory=DBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


CONFIG = PipelineConfig()