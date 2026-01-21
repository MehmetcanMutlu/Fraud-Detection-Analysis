import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- AYARLAR ---
st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide")

# --- MODEL YÜKLEME (Cache ile hızlandırılmış) ---
@st.cache_resource
def load_artifacts():
    xgb_model = joblib.load('model_xgb.pkl')
    lgb_model = joblib.load('model_lgb.pkl')
    demo_data = joblib.load('demo_data.pkl')
    return xgb_model, lgb_model, demo_data

try:
    model_xgb, model_lgb, demo_data = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model dosyaları bulunamadı! Lütfen önce 'save_model.py' dosyasını çalıştırın.")
    st.stop()

# --- SIDEBAR (KONTROL PANELİ) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2058/2058768.png", width=100)
st.sidebar.title("🛡️ FraudGuard AI")
st.sidebar.markdown("---")
st.sidebar.write("Bu panel, **XGBoost** ve **LightGBM** hibrit modeli kullanarak işlemleri analiz eder.")

# İşlem Seçimi
st.sidebar.subheader("🔎 İşlem Analizi")
random_transaction = st.sidebar.button("🎲 Rastgele İşlem Seç")

# Demo verisinden rastgele bir satır seç
if 'selected_idx' not in st.session_state or random_transaction:
    st.session_state.selected_idx = np.random.choice(demo_data.index)

selected_row = demo_data.loc[[st.session_state.selected_idx]]
transaction_id = st.session_state.selected_idx # ID olarak index'i kullanıyoruz demo için

st.sidebar.info(f"Seçilen İşlem ID: **{transaction_id}**")

# --- ANA EKRAN ---
st.title("Finansal Güvenlik Paneli")
st.markdown("Gerçek zamanlı dolandırıcılık tespit sistemi analizi.")

col1, col2 = st.columns([2, 1])

# --- TAHMİN MEKANİZMASI ---
pred_xgb = model_xgb.predict_proba(selected_row)[0][1]
pred_lgb = model_lgb.predict_proba(selected_row)[0][1]
final_prob = (0.5 * pred_xgb) + (0.5 * pred_lgb) # Ensemble

# --- GRAFİK (GAUGE CHART) ---
fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = final_prob * 100,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Dolandırıcılık Riski (%)", 'font': {'size': 24}},
    delta = {'reference': 50, 'increasing': {'color': "red"}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 20], 'color': "#00ff00"},  # Yeşil (Güvenli)
            {'range': [20, 50], 'color': "#ffff00"}, # Sarı (Şüpheli)
            {'range': [50, 100], 'color': "#ff0000"}], # Kırmızı (Tehlikeli)
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': 50}}))

with col1:
    st.plotly_chart(fig, use_container_width=True)

# --- KARAR MEKANİZMASI ---
with col2:
    st.subheader("📋 Analiz Sonucu")
    if final_prob > 0.50:
        st.error("🚨 DİKKAT: YÜKSEK RİSK!")
        st.write("Bu işlem büyük ihtimalle **DOLANDIRICILIK**.")
        st.markdown(f"**Güven Skoru:** %{100 - (final_prob*100):.2f}")
    elif final_prob > 0.20:
        st.warning("⚠️ UYARI: ŞÜPHELİ İŞLEM")
        st.write("İnceleme yapılması önerilir.")
    else:
        st.success("✅ GÜVENLİ İŞLEM")
        st.write("Herhangi bir risk tespit edilmedi.")

    st.markdown("---")
    st.write("🤖 **Model Görüşleri:**")
    st.write(f"- XGBoost: %{pred_xgb*100:.2f}")
    st.write(f"- LightGBM: %{pred_lgb*100:.2f}")

# --- DETAY TABLOSU ---
st.markdown("---")
st.subheader("📊 İşlem Detayları (Ham Veri)")
st.dataframe(selected_row)

# --- FOOTER ---
st.markdown("---")
st.caption("Developed by Mehmetcan | Powered by Streamlit & XGBoost/LightGBM Ensemble")