# 🕵️‍♂️ IEEE-CIS Fraud Detection Project

## 📌 Proje Hakkında
Bu proje, finansal işlemlerde kredi kartı dolandırıcılığını (fraud) tespit etmek amacıyla geliştirilmiştir. IEEE-CIS veri seti kullanılarak, milyonlarca işlem arasından dolandırıcıları yakalayan bir Makine Öğrenmesi modeli kurulmuştur.

Büyük ölçekli ve dengesiz (imbalanced) veri setleri üzerinde çalışılmış, **Feature Engineering** teknikleri ile ham veri anlamlı hale getirilmiştir.

## 🏆 Kaggle Başarı Skorları
Modelin Kaggle yarışmasındaki gerçek performans sonuçları:
* **Public Score:** `0.9102` 🚀
* **Private Score:** `0.8528`
* **Local ROC-AUC:** `0.9416`

## 📊 Modelin "Dolandırıcıyı" Yakalama Kriterleri (Feature Importance)
Modelimiz, işlemleri analiz ederken aşağıdaki özelliklere dikkat etmektedir. Grafikte dolandırıcılığı en çok ele veren faktörler görülmektedir:

![Feature Importance Grafiği](outputs/feature_importance.png)

## 🛠️ Kullanılan Teknolojiler ve Yöntemler
* **Python:** Ana geliştirme dili
* **XGBoost:** Sınıflandırma modeli
* **Pandas & NumPy:** Veri işleme ve manipülasyon
* **Matplotlib & Seaborn:** Veri görselleştirme
* **İleri Seviye Teknikler:**
    * **Data Leakage Prevention:** Time-series split ve Index Slicing yöntemleri
    * **Class Imbalance Handling:** `scale_pos_weight` optimizasyonu
    * **Modular Architecture:** `src/` klasörü altında temiz kod yapısı

## 🚀 Kurulum
Projeyi kendi bilgisayarınızda çalıştırmak için:

```bash
# 1. Repoyu klonlayın
git clone [https://github.com/MehmetcanMutlu/Fraud-Detection-Analysis.git](https://github.com/MehmetcanMutlu/Fraud-Detection-Analysis.git)

# 2. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# 3. Kaggle veri setlerini 'input' klasörüne indirin.

# 4. Modeli eğitin ve tahmin üretin
python3 submission.py