"""
agent4_visualizer.py
----------------------
Ajan 4: UI/UX & Visualization (Kurumsal Seviye - TradingView Stili)

Sorumluluk:
  1) 3 panelli (subplot) Plotly grafiği:
     - Panel 1: Mum grafiği + Fibonacci + Destek/Direnç + VWAP + AI Al/Sat sinyalleri
     - Panel 2: Renklendirilmiş Hacim çubukları + OBV çizgisi
     - Panel 3: RSI + ADX (0-100 ortak eksen) + MACD (ikincil eksen, histogram)
     "plotly_dark" tema + silikleştirilmiş grid ile TradingView hissiyatı.
  2) Flutter / mobil entegrasyon için temiz bir JSON sinyal çıktısı üretmek.
  3) Kural tabanlı, açıklanabilir bir "teknik değerlendirme" (alım fırsatı mı?)
     metni üretmek.
"""

import json
import logging
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CONFIG

logger = logging.getLogger(__name__)

# TradingView'e yakın renk paleti
COLOR_UP = "#26a69a"
COLOR_DOWN = "#ef5350"
COLOR_VOLUME_UP = "rgba(38, 166, 154, 0.6)"
COLOR_VOLUME_DOWN = "rgba(239, 83, 80, 0.6)"
COLOR_GRID = "rgba(255,255,255,0.06)"
COLOR_TREND = "#42a5f5"
COLOR_VWAP = "#ab47bc"
COLOR_OBV = "#ffca28"


class VisualizationAgent:
    """Model çıktısını profesyonel görünümlü grafiğe, JSON'a ve
    kural tabanlı teknik yoruma çeviren ajan."""

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Yardımcı: basit doğrusal regresyon trend çizgisi
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_trend_line(df: pd.DataFrame) -> np.ndarray:
        x = np.arange(len(df))
        y = df["close"].values
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            return np.full(len(df), np.nan)
        coeffs = np.polyfit(x[valid], y[valid], deg=1)
        return np.polyval(coeffs, x)

    # ------------------------------------------------------------------
    # Ana grafik: 3 panelli kurumsal görünüm
    # ------------------------------------------------------------------
    def plot_candlestick(self, df: pd.DataFrame, symbol: str, save_html: bool = True,
                          display_window: int = 180) -> go.Figure:
        df = df.sort_values("date").reset_index(drop=True)
        latest = df.iloc[-1]
        plot_df = df.tail(display_window).copy().reset_index(drop=True)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.18, 0.22],
            vertical_spacing=0.02,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]],
        )

        # ================= PANEL 1: Fiyat =================
        fig.add_trace(go.Candlestick(
            x=plot_df["date"], open=plot_df["open"], high=plot_df["high"],
            low=plot_df["low"], close=plot_df["close"], name=symbol,
            increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN,
            increasing_fillcolor=COLOR_UP, decreasing_fillcolor=COLOR_DOWN,
        ), row=1, col=1)

        trend_y = self._compute_trend_line(plot_df)
        trend_direction = "Yükseliş" if trend_y[-1] > trend_y[0] else "Düşüş"
        fig.add_trace(go.Scatter(
            x=plot_df["date"], y=trend_y, name=f"Trend ({trend_direction})",
            line=dict(color=COLOR_TREND, width=2),
        ), row=1, col=1)

        if "vwap" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df["date"], y=plot_df["vwap"], name=f"VWAP ({CONFIG.vwap_window}g)",
                line=dict(color=COLOR_VWAP, width=1.5, dash="dot"),
            ), row=1, col=1)

        if pd.notna(latest.get("resistance_50d")):
            fig.add_hline(y=latest["resistance_50d"], line_dash="dot", line_color=COLOR_DOWN,
                          line_width=1.2, annotation_text=f"Direnç {latest['resistance_50d']:.2f}",
                          annotation_position="top left", annotation_font_color=COLOR_DOWN, row=1, col=1)
        if pd.notna(latest.get("support_50d")):
            fig.add_hline(y=latest["support_50d"], line_dash="dot", line_color=COLOR_UP,
                          line_width=1.2, annotation_text=f"Destek {latest['support_50d']:.2f}",
                          annotation_position="bottom left", annotation_font_color=COLOR_UP, row=1, col=1)
        if pd.notna(latest.get("fib_618")):
            fig.add_hline(y=latest["fib_618"], line_dash="dash", line_color="#ffd54f",
                          line_width=1.2, annotation_text=f"Fib 0.618 {latest['fib_618']:.2f}",
                          annotation_position="top right", annotation_font_color="#ffd54f", row=1, col=1)

        if "ml_prediction" in plot_df.columns:
            plot_df["prev_pred"] = plot_df["ml_prediction"].shift(1)
            changes = plot_df[plot_df["ml_prediction"] != plot_df["prev_pred"]]
            buy_points = changes[changes["ml_prediction"] == 1]
            sell_points = changes[changes["ml_prediction"] == -1]

            fig.add_trace(go.Scatter(
                x=buy_points["date"], y=buy_points["low"] * 0.97, mode="markers",
                name="Al sinyali", marker=dict(symbol="triangle-up", size=13, color=COLOR_UP,
                                                 line=dict(width=1, color="white")),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=sell_points["date"], y=sell_points["high"] * 1.03, mode="markers",
                name="Sat sinyali", marker=dict(symbol="triangle-down", size=13, color=COLOR_DOWN,
                                                  line=dict(width=1, color="white")),
            ), row=1, col=1)

        # ================= PANEL 2: Hacim + OBV =================
        vol_colors = [COLOR_VOLUME_UP if c >= o else COLOR_VOLUME_DOWN
                      for o, c in zip(plot_df["open"], plot_df["close"])]
        fig.add_trace(go.Bar(
            x=plot_df["date"], y=plot_df["volume"], name="Hacim",
            marker_color=vol_colors, showlegend=True,
        ), row=2, col=1)

        if "obv" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df["date"], y=plot_df["obv"], name="OBV",
                line=dict(color=COLOR_OBV, width=1.5), yaxis="y2",
            ), row=2, col=1)

        # ================= PANEL 3: RSI + ADX (ortak 0-100 ekseni) + MACD (ikincil eksen) =================
        if "rsi_14" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df["date"], y=plot_df["rsi_14"], name="RSI(14)",
                line=dict(color="#29b6f6", width=1.5),
            ), row=3, col=1, secondary_y=False)
        if "adx_14" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df["date"], y=plot_df["adx_14"], name="ADX(14) - trend gücü",
                line=dict(color="#ffa726", width=1.5, dash="dot"),
            ), row=3, col=1, secondary_y=False)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.4)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.4)", row=3, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="rgba(255,167,38,0.4)", row=3, col=1)

        if "macd_hist" in plot_df.columns:
            macd_colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in plot_df["macd_hist"].fillna(0)]
            fig.add_trace(go.Bar(
                x=plot_df["date"], y=plot_df["macd_hist"], name="MACD Hist",
                marker_color=macd_colors, opacity=0.5,
            ), row=3, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(
                x=plot_df["date"], y=plot_df["macd"], name="MACD",
                line=dict(color="#7e57c2", width=1.2),
            ), row=3, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(
                x=plot_df["date"], y=plot_df["macd_signal"], name="Sinyal",
                line=dict(color="#ec407a", width=1.2, dash="dot"),
            ), row=3, col=1, secondary_y=True)

        # ================= Bilgi kutusu =================
        info_lines = [f"<b>{symbol} — Güncel Seviyeler</b>"]
        info_lines.append(f"Kapanış: {latest['close']:.2f}")
        for label, key, fmt in [
            ("Destek (50g)", "support_50d", "{:.2f}"), ("Direnç (50g)", "resistance_50d", "{:.2f}"),
            ("Fib 0.618", "fib_618", "{:.2f}"), ("VWAP", "vwap", "{:.2f}"),
            ("RSI(14)", "rsi_14", "{:.1f}"), ("ADX(14)", "adx_14", "{:.1f}"),
            ("ATR(14)", "atr_14", "{:.2f}"),
        ]:
            val = latest.get(key)
            if pd.notna(val):
                info_lines.append(f"{label}: {fmt.format(val)}")

        fig.add_annotation(
            xref="paper", yref="paper", x=0.995, y=0.99, xanchor="right", yanchor="top",
            text="<br>".join(info_lines), showarrow=False, align="left",
            bgcolor="rgba(30,30,30,0.85)", bordercolor="rgba(255,255,255,0.2)", borderwidth=1,
            font=dict(size=11, color="white"),
        )

        # ================= Tema: plotly_dark + silik grid =================
        fig.update_layout(
            template="plotly_dark",
            title=f"{symbol} - Kurumsal Teknik Analiz (son {len(plot_df)} gün)",
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#131722", plot_bgcolor="#131722",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
            margin=dict(l=50, r=50, t=60, b=30),
            height=900,
        )
        fig.update_xaxes(gridcolor=COLOR_GRID, showspikes=True, spikemode="across", spikecolor="grey")
        fig.update_yaxes(gridcolor=COLOR_GRID)
        fig.update_yaxes(title_text="Fiyat", row=1, col=1)
        fig.update_yaxes(title_text="Hacim / OBV", row=2, col=1)
        fig.update_yaxes(title_text="RSI / ADX", range=[0, 100], row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="MACD", row=3, col=1, secondary_y=True, showgrid=False)

        if save_html:
            path = f"{self.output_dir}/{symbol}_analysis.html"
            fig.write_html(path)
            logger.info(f"Grafik kaydedildi: {path}")

        return fig

    # ------------------------------------------------------------------
    # Çıktı 2: Flutter / mobil entegrasyon için JSON API çıktısı
    # ------------------------------------------------------------------
    def get_latest_signal_json(self, df: pd.DataFrame, symbol: str,
                                fundamental_analysis: dict = None) -> str:
        """
        fundamental_analysis: Ajan 5'ten (FundamentalAnalystAgent.run()) dönen
        {'ratios': {...}, 'llm_report': '...'} sözlüğü. None geçilirse JSON'da
        'fundamental_analysis' alanı da None olarak yer alır (Flutter tarafı
        bu durumda "temel analiz mevcut değil" olarak ele almalı).
        """
        if df.empty:
            return json.dumps({"error": f"{symbol} için veri bulunamadı"}, ensure_ascii=False)

        latest = df.sort_values("date").iloc[-1]

        signal_map = {1: "UCUZ_AL_FIRSATI", -1: "PAHALI_SAT_FIRSATI", 0: "YATAY_NOTR"}
        prediction_raw = latest.get("ml_prediction")
        prediction_label = signal_map.get(
            int(prediction_raw) if pd.notna(prediction_raw) else 0, "YATAY_NOTR"
        )

        payload = {
            "symbol": symbol,
            "date": latest["date"].strftime("%Y-%m-%d") if pd.notna(latest["date"]) else None,
            "price": {
                "open": round(float(latest["open"]), 2),
                "high": round(float(latest["high"]), 2),
                "low": round(float(latest["low"]), 2),
                "close": round(float(latest["close"]), 2),
            },
            "indicators": {
                "rsi_14": self._safe_round(latest.get("rsi_14")),
                "macd": self._safe_round(latest.get("macd")),
                "stoch_k": self._safe_round(latest.get("stoch_k")),
                "atr_14": self._safe_round(latest.get("atr_14")),
                "adx_14": self._safe_round(latest.get("adx_14")),
                "vwap": self._safe_round(latest.get("vwap")),
                "obv": self._safe_round(latest.get("obv"), 0),
            },
            "levels": {
                "support_50d": self._safe_round(latest.get("support_50d")),
                "resistance_50d": self._safe_round(latest.get("resistance_50d")),
                "fib_618": self._safe_round(latest.get("fib_618")),
                "dist_to_fib_618_pct": self._safe_round(latest.get("dist_to_fib_618")),
            },
            "signal": {
                "code": int(prediction_raw) if pd.notna(prediction_raw) else 0,
                "label": prediction_label,
            },
            "fundamental_analysis": fundamental_analysis if fundamental_analysis else None,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Çıktı 3: Kural tabanlı, açıklanabilir teknik değerlendirme
    # ------------------------------------------------------------------
    def generate_technical_assessment(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        ML modelinden BAĞIMSIZ, saydam kural tabanlı bir teknik puanlama.
        Artık ADX ile "trend var mı yok mu" bilgisini de dikkate alıyor:
        ADX < 20 iken üretilen Al/Sat sinyalleri "yatay piyasada whipsaw
        riski" notuyla etiketleniyor.
        """
        latest = df.sort_values("date").iloc[-1]
        score = 0
        reasons = []

        adx = latest.get("adx_14")
        trending_market = True
        if pd.notna(adx):
            if adx < 20:
                trending_market = False
                reasons.append(f"ADX {adx:.1f} (<20) — piyasa YATAY, trend sinyalleri güvenilirliği düşük.")
            elif adx > 25:
                reasons.append(f"ADX {adx:.1f} (>25) — belirgin bir TREND mevcut.")
            else:
                reasons.append(f"ADX {adx:.1f} — trend gücü orta seviyede.")

        rsi = latest.get("rsi_14")
        if pd.notna(rsi):
            if rsi < 35:
                score += 1
                reasons.append(f"RSI {rsi:.1f} ile aşırı satım bölgesine yakın (bullish).")
            elif rsi > 65:
                score -= 1
                reasons.append(f"RSI {rsi:.1f} ile aşırı alım bölgesine yakın (bearish).")
            else:
                reasons.append(f"RSI {rsi:.1f} ile nötr bölgede.")

        close = latest["close"]
        support = latest.get("support_50d")
        resistance = latest.get("resistance_50d")
        if pd.notna(support) and pd.notna(resistance) and resistance > support:
            range_pos = (close - support) / (resistance - support)
            if range_pos <= 0.15:
                score += 1
                reasons.append("Fiyat 50 günlük DESTEK seviyesine çok yakın (bullish).")
            elif range_pos >= 0.85:
                score -= 1
                reasons.append("Fiyat 50 günlük DİRENÇ seviyesine çok yakın (bearish).")
            else:
                reasons.append(f"Fiyat destek-direnç aralığının %{range_pos*100:.0f} noktasında.")

        vwap = latest.get("vwap")
        if pd.notna(vwap):
            if close > vwap:
                reasons.append(f"Fiyat VWAP'ın ({vwap:.2f}) ÜZERİNDE — kısa vadeli alıcılar kontrolde.")
            else:
                reasons.append(f"Fiyat VWAP'ın ({vwap:.2f}) ALTINDA — kısa vadeli satıcılar kontrolde.")

        dist_fib = latest.get("dist_to_fib_618")
        if pd.notna(dist_fib) and abs(dist_fib) <= 3:
            score += 1
            reasons.append(f"Fiyat Fibonacci 0.618 (Golden Zone) seviyesine %{dist_fib:.1f} mesafede.")

        macd_hist = latest.get("macd_hist")
        if pd.notna(macd_hist):
            if macd_hist > 0:
                score += 1
                reasons.append("MACD histogramı pozitif (momentum yukarı yönlü).")
            else:
                score -= 1
                reasons.append("MACD histogramı negatif (momentum aşağı yönlü).")

        ml_pred = latest.get("ml_prediction")
        whipsaw_warning = False
        if pd.notna(ml_pred):
            if not trending_market and ml_pred != 0:
                whipsaw_warning = True
                reasons.append("⚠ ML sinyali YATAY piyasada üretildi — whipsaw (sahte sinyal) riski yüksek.")
            if ml_pred == 1:
                score += 1
                reasons.append("ML modeli 'Ucuz/Alım Fırsatı' etiketi üretti.")
            elif ml_pred == -1:
                score -= 1
                reasons.append("ML modeli 'Pahalı/Satış Fırsatı' etiketi üretti.")
            else:
                reasons.append("ML modeli 'Yatay/Nötr' etiketi üretti.")

        if score >= 3:
            verdict = "GÜÇLÜ ALIM BÖLGESİ"
        elif score == 2:
            verdict = "ALIM İÇİN İZLENEBİLİR"
        elif score in (0, 1):
            verdict = "NÖTR / KARARSIZ"
        elif score == -2:
            verdict = "TEMKİNLİ / ZAYIF SATIŞ SİNYALİ"
        else:
            verdict = "PAHALI / SATIŞ BÖLGESİ"

        if whipsaw_warning:
            verdict += " (DİKKAT: Yatay Piyasa)"

        return {
            "symbol": symbol,
            "date": latest["date"].strftime("%Y-%m-%d") if pd.notna(latest["date"]) else None,
            "score": score,
            "verdict": verdict,
            "trending_market": trending_market,
            "reasons": reasons,
            "disclaimer": "Bu değerlendirme kural tabanlı bir teknik analiz özetidir, yatırım tavsiyesi değildir.",
        }

    def print_assessment(self, assessment: dict):
        print(f"\n{'='*55}")
        print(f"  {assessment['symbol']} TEKNİK DEĞERLENDİRME ({assessment['date']})")
        print(f"{'='*55}")
        print(f"  SONUÇ: {assessment['verdict']}  (puan: {assessment['score']})")
        print(f"{'-'*55}")
        for r in assessment["reasons"]:
            print(f"  • {r}")
        print(f"{'-'*55}")
        print(f"  ⚠ {assessment['disclaimer']}")
        print(f"{'='*55}\n")

    @staticmethod
    def _safe_round(value, digits=2):
        return round(float(value), digits) if pd.notna(value) else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)