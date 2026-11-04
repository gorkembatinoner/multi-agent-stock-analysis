"""
orchestrator.py (main.py olarak da çalıştırılabilir)
------------------------------------------------------
TradingPipelineOrchestrator: 4 ajanı sırayla tetikleyip aralarındaki
veri akışını (DataFrame + DB) yöneten üst düzey sınıf.

Akış:
  Agent1 (fetch + DB'ye upsert)
     -> Agent2 (DB'den oku, feature ekle, DB'ye geri yaz)
        -> Agent3 (feature'lı df ile etiketle + eğit + tahmin et)
           -> Agent4 (grafik çiz + JSON üret)
"""

import json
import logging

from config import CONFIG
from database import DatabaseManager
from agent1_data_collector import DataCollectorAgent
from agent2_feature_engineer import FeatureEngineerAgent
from agent3_ml_engineer import MLEngineerAgent
from agent4_visualizer import VisualizationAgent
from agent5_fundamental_analyst import FundamentalAnalystAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Orchestrator")


class TradingPipelineOrchestrator:
    """4 ajanı yöneten, tek bir noktadan pipeline'ı çalıştıran sınıf."""

    def __init__(self, output_dir: str = "."):
        self.db_manager = DatabaseManager()
        self.collector = DataCollectorAgent(self.db_manager)
        self.feature_engineer = FeatureEngineerAgent(self.db_manager)
        self.ml_engineer = MLEngineerAgent()
        self.visualizer = VisualizationAgent(output_dir=output_dir)
        self.fundamental_analyst = FundamentalAnalystAgent()

        self.latest_signals: dict[str, str] = {}  # symbol -> JSON string

    def run_full_pipeline(self, symbols=None):
        symbols = symbols or CONFIG.symbols
        logger.info(f"Pipeline başlıyor. Semboller: {symbols}")

        # 1) VERİ TOPLAMA
        logger.info("=== ADIM 1: Agent 1 - Veri Toplama ===")
        self.collector.run(symbols)

        for symbol in symbols:
            try:
                self._run_single_symbol(symbol)
            except Exception as e:
                logger.error(f"{symbol} pipeline'ı başarısız oldu: {e}")

        logger.info("Pipeline tamamlandı.")
        return self.latest_signals

    def _run_single_symbol(self, symbol: str):
        # 2) FEATURE ENGINEERING
        logger.info(f"=== ADIM 2: Agent 2 - Feature Engineering ({symbol}) ===")
        features_df = self.feature_engineer.run(symbol)
        if features_df.empty:
            logger.warning(f"{symbol}: feature verisi boş, atlanıyor.")
            return

        # 3) MODEL EĞİTİMİ + TAHMİN
        logger.info(f"=== ADIM 3: Agent 3 - ML Eğitim & Tahmin ({symbol}) ===")
        metrics = self.ml_engineer.train(features_df)
        logger.info(f"{symbol}: eğitim tamam, {metrics['n_samples']} örnek kullanıldı.")
        predicted_df = self.ml_engineer.predict(features_df)

        # 4) GÖRSELLEŞTİRME + JSON + TEKNİK DEĞERLENDİRME
        logger.info(f"=== ADIM 4: Agent 4 - Görselleştirme ({symbol}) ===")
        self.visualizer.plot_candlestick(predicted_df, symbol, save_html=True)

        # 5) TEMEL ANALİZ (Ajan 5) -- teknik pipeline'dan BAĞIMSIZ, ayrı veri
        # kaynağı (yfinance fundamentals) kullanır; başarısız olsa bile Agent
        # 4'ün JSON çıktısını engellemez (fundamental_analysis=None kalır).
        logger.info(f"=== ADIM 5: Agent 5 - Temel Analiz ({symbol}) ===")
        try:
            fundamental_result = self.fundamental_analyst.run(symbol)
            logger.info(f"{symbol} temel analiz rasyoları: {fundamental_result['ratios']}")
            logger.info(f"{symbol} LLM raporu:\n{fundamental_result['llm_report']}")
        except Exception as e:
            logger.error(f"{symbol} temel analiz başarısız oldu: {e}")
            fundamental_result = None

        signal_json = self.visualizer.get_latest_signal_json(
            predicted_df, symbol, fundamental_analysis=fundamental_result
        )
        self.latest_signals[symbol] = signal_json
        logger.info(f"{symbol} güncel sinyal:\n{signal_json}")

        assessment = self.visualizer.generate_technical_assessment(predicted_df, symbol)
        self.visualizer.print_assessment(assessment)

    def get_all_signals_json(self) -> str:
        """Tüm sembollerin en güncel sinyallerini tek bir JSON'da toplar (Flutter için)."""
        parsed = {sym: json.loads(js) for sym, js in self.latest_signals.items()}
        return json.dumps(parsed, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import sys

    orchestrator = TradingPipelineOrchestrator(output_dir="./outputs")

    # Terminalden hisse kodu al: "python orchestrator.py AAPL" şeklinde
    # argüman verilmişse onu kullan, verilmemişse interaktif olarak sor.
    if len(sys.argv) > 1:
        symbol_input = sys.argv[1].strip().upper()
    else:
        symbol_input = input("Analiz edilecek hisse kodunu girin (örn. AAPL): ").strip().upper()

    if not symbol_input:
        print("Geçerli bir hisse kodu girmediniz, çıkılıyor.")
        sys.exit(1)

    orchestrator.run_full_pipeline(symbols=[symbol_input])

    print(f"\n=== {symbol_input} SİNYALİ (Flutter API çıktısı) ===")
    print(orchestrator.get_all_signals_json())