"""
agent3_ml_engineer.py
-----------------------
Ajan 3: Machine Learning Engineer (Model Eğitimi ve Tahmin)

Sorumluluk: Feature'lı veriyi etiketlemek (target variable), zaman serisine
uygun (asla random olmayan) train/test ayrımıyla XGBoost sınıflandırıcı eğitmek,
model performansını raporlamak ve tahmin üretmek.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import CONFIG

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "stoch_k", "stoch_d",
    "atr_14", "vwap", "adx_14", "obv",
    "support_50d", "resistance_50d", "fib_618", "dist_to_fib_618",
]


class MLEngineerAgent:
    """Etiketleme + zaman serisi duyarlı model eğitimi yapan ajan."""

    def __init__(self):
        self.model: XGBClassifier | None = None
        self.label_map = {-1: 0, 0: 1, 1: 2}          # dahili sabit gösterim
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        # Üretim modelinin fiilen gördüğü sınıflar için LabelEncoder.
        # XGBoost, etiketlerin 0'dan başlayıp ARALIKSIZ olmasını şart koşar;
        # ATR-tabanlı dinamik etiketleme bazı dönemlerde (örn. güçlü tek
        # yönlü trend) 3 sınıftan sadece 2'sini üretebilir (örn. sadece
        # {0, 2} gibi) -> bu encoder, hangi sınıflar fiilen mevcutsa onları
        # 0..k-1 aralığına yeniden kodlar, tahminler geri orijinale çevrilir.
        self.production_encoder: LabelEncoder | None = None

    # ------------------------------------------------------------------
    # Etiketleme (Target Variable)
    # ------------------------------------------------------------------
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DİNAMİK (ATR tabanlı) etiketleme kuralı:
          - Sonraki N (label_horizon_days) işlem gününde fiyat, O GÜNKÜ ATR
            değerinin (label_atr_multiplier) katından fazla ARTARSA -> 1
            (Ucuz / Alım Fırsatı)
          - Aynı miktarda DÜŞERSE -> -1 (Pahalı / Satış Fırsatı)
          - Aksi halde -> 0 (Yatay)

        Sabit yüzde eşiği yerine ATR kullanmanın nedeni: durgun (düşük
        oynaklıklı) bir hissede %3 aylarca hiç tetiklenmeyebilirken, oynak
        bir hissede günlük normal dalgalanmayla bile tetiklenip gürültülü
        etiketler üretebilir. ATR, hissenin KENDİ ortalama günlük hareketine
        göre orantılı bir hedef belirler.

        NOT: Bu ileriye bakan (future) bir hesaplamadır, sadece eğitim
        etiketi üretmek içindir; son `horizon` satırda NaN kalır ve
        eğitimden çıkarılır (aşağıda dropna).
        """
        df = df.copy()
        horizon = CONFIG.label_horizon_days
        multiplier = CONFIG.label_atr_multiplier

        future_price = df["close"].shift(-horizon)
        future_change = future_price - df["close"]
        dynamic_target = df["atr_14"] * multiplier  # o günkü ATR'ye göre hedef

        conditions = [
            future_change > dynamic_target,
            future_change < -dynamic_target,
        ]
        choices = [1, -1]
        df["ml_label"] = np.select(conditions, choices, default=0)

        # future_price veya ATR NaN ise etiket anlamsızdır
        invalid_mask = future_price.isna() | dynamic_target.isna()
        df.loc[invalid_mask, "ml_label"] = np.nan
        return df

    # ------------------------------------------------------------------
    # Time-Series duyarlı train/test ayrımı (KESİNLİKLE random değil)
    # ------------------------------------------------------------------
    def time_series_split(self, X: pd.DataFrame, y: pd.Series):
        """
        sklearn TimeSeriesSplit -> her fold'da train seti test setinden
        HER ZAMAN kronolojik olarak önce gelir. Look-ahead bias engellenir.
        Son fold, "gerçek" nihai test seti olarak kullanılır.
        """
        tscv = TimeSeriesSplit(n_splits=CONFIG.n_splits)
        splits = list(tscv.split(X))
        return splits  # [(train_idx, test_idx), ...] kronolojik sırayla artan

    # ------------------------------------------------------------------
    # Eğitim
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame) -> dict:
        df = self.create_labels(df)
        df_train_ready = df.dropna(subset=FEATURE_COLUMNS + ["ml_label"]).reset_index(drop=True)

        if len(df_train_ready) < 100:
            raise ValueError("Eğitim için yeterli temiz veri yok (min ~100 satır önerilir).")

        X = df_train_ready[FEATURE_COLUMNS]
        y = df_train_ready["ml_label"].map(self.label_map)  # -1,0,1 -> 0,1,2

        label_dist = df_train_ready["ml_label"].value_counts(normalize=True).round(3) * 100
        logger.info(f"Etiket dağılımı: {dict(label_dist)}")

        splits = self.time_series_split(X, y)

        # Cross-validation raporu: her fold kronolojik olarak ilerler
        fold_reports = []
        for i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # ATR-tabanlı dinamik etiketleme, güçlü tek yönlü trendlerde bazı
            # fold'larda sadece TEK sınıf bırakabiliyor (örn. hep "Al" etiketi).
            # XGBoost 3 sınıflı model 2+ sınıf görmeden eğitilemez -> bu fold'u
            # atla, pipeline'ın tamamı çökmesin.
            if y_train.nunique() < 2:
                logger.warning(
                    f"Fold {i+1}/{len(splits)} atlandı: eğitim setinde sadece "
                    f"{y_train.nunique()} sınıf var (yetersiz çeşitlilik)."
                )
                continue

            model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                random_state=42,
            )
            # XGBoost etiketlerin 0'dan başlayıp aralıksız olmasını şart koşar;
            # bu fold'da fiilen görünen sınıfları (örn. {1,2}) 0..k-1'e (örn.
            # {0,1}) yeniden kodlayıp, tahminleri orijinal etikete geri çeviriyoruz.
            fold_encoder = LabelEncoder()
            y_train_enc = fold_encoder.fit_transform(y_train)

            # Sınıf ağırlıklandırması: "Yatay" etiketi çoğunlukta olduğu için
            # ağırlıksız eğitimde model çoğunlukla onu tahmin etmeye kayar.
            # 'balanced' ağırlık, azınlık sınıflara (Al/Sat) orantılı olarak
            # daha fazla önem verir -> model gerçek Al/Sat kalıplarını öğrenmeye zorlanır.
            sample_weights = compute_sample_weight("balanced", y_train_enc)
            model.fit(X_train, y_train_enc, sample_weight=sample_weights)
            preds_enc = model.predict(X_test)
            preds = fold_encoder.inverse_transform(preds_enc)

            report = classification_report(
                y_test, preds, labels=[0, 1, 2], output_dict=True, zero_division=0
            )
            fold_reports.append(report)
            logger.info(f"Fold {i+1}/{len(splits)} - accuracy: {report['accuracy']:.3f}")

        if not fold_reports:
            logger.warning("Hiçbir fold değerlendirilemedi (sınıf çeşitliliği yetersiz).")

        # Nihai (production) model: TÜM veriyle eğitilir.
        if y.nunique() < 2:
            raise ValueError(
                "Tüm veri setinde tek bir sınıf var, model eğitilemez. "
                "label_atr_multiplier değerini düşürmeyi deneyin (config.py)."
            )

        # Nihai (production) model: TÜM veriyle eğitilir. Aynı LabelEncoder
        # mantığı burada da uygulanır ve predict() tarafında kullanılmak
        # üzere self.production_encoder olarak saklanır.
        self.production_encoder = LabelEncoder()
        y_encoded = self.production_encoder.fit_transform(y)
        self.model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42,
        )
        production_weights = compute_sample_weight("balanced", y_encoded)
        self.model.fit(X, y_encoded, sample_weight=production_weights)

        return {"fold_reports": fold_reports, "n_samples": len(df_train_ready)}

    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model henüz eğitilmedi. Önce train() çağırın.")

        df = df.copy()
        valid_mask = df[FEATURE_COLUMNS].notna().all(axis=1)
        preds = pd.Series(np.nan, index=df.index)

        if valid_mask.any():
            X_valid = df.loc[valid_mask, FEATURE_COLUMNS]
            raw_preds_enc = self.model.predict(X_valid)
            # Önce production_encoder ile orijinal 0/1/2 kodlamasına geri
            # dön, sonra inverse_label_map ile gerçek -1/0/1 etiketine çevir.
            raw_preds = self.production_encoder.inverse_transform(raw_preds_enc)
            preds.loc[valid_mask] = [self.inverse_label_map[p] for p in raw_preds]

        df["ml_prediction"] = preds
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from agent2_feature_engineer import FeatureEngineerAgent

    fe_agent = FeatureEngineerAgent()
    ml_agent = MLEngineerAgent()

    sample_symbol = CONFIG.symbols[0]
    features_df = fe_agent.run(sample_symbol)
    metrics = ml_agent.train(features_df)
    predicted_df = ml_agent.predict(features_df)
    print(predicted_df[["date", "close", "ml_prediction"]].tail(10))