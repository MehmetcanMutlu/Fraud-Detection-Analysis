import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np

def train_model(df):
    """
    XGBoost modelini eğitir ve test eder.
    """
    print("🤖 Model Eğitimi Başlıyor (XGBoost)...")

    # 1. Hedef ve Özellikleri Ayır
    # isFraud: Hedefimiz (y)
    # TransactionID: Tahmin için gereksiz (Sadece sıra numarası), siliyoruz.
    y = df['isFraud']
    X = df.drop(['isFraud', 'TransactionID'], axis=1)
    
    # 2. Train / Test Ayrımı
    # Verinin %80'i ile ders çalışacak (Train), %20'si ile sınava girecek (Test).
    # stratify=y -> Fraud oranı (%3.5) hem eğitimde hem testte aynı kalsın diye.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Eğitim Seti: {X_train.shape}, Test Seti: {X_test.shape}")

    # 3. Dengesizlik Ayarı (Class Imbalance)
    # Normal işlemlerin Fraud işlemlere oranı. Model bu sayıyı kullanıp Fraud'a odaklanacak.
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    print(f"[INFO] Dengesizlik Oranı (scale_pos_weight): {ratio:.2f}")

    # 4. Modeli Kur
    model = xgb.XGBClassifier(
        n_estimators=50,        # Ağaç sayısı (Hızlı olsun diye 50, normalde 500+ yapılır)
        max_depth=10,           # Ağaç derinliği
        learning_rate=0.1,      # Öğrenme hızı
        scale_pos_weight=ratio, # KRİTİK AYAR!
        eval_metric='auc',      # Başarı kriterimiz AUC (Doğruluk değil!)
        use_label_encoder=False,
        random_state=42
    )

    # 5. Modeli Eğit (Fit)
    model.fit(X_train, y_train)
    
    # 6. Tahmin Yap (Sınav)
    # predict_proba -> Bize 0 veya 1 değil, "Fraud olma ihtimalini" (Örn: %85) verir.
    preds_prob = model.predict_proba(X_test)[:, 1]
    
    # 7. Performansı Ölç (ROC-AUC)
    auc_score = roc_auc_score(y_test, preds_prob)
    print(f"\n🏆 TEST SONUCU (ROC-AUC Skoru): %{auc_score * 100:.2f}")
    
    # 0.5 = Yazı Tura (Berbat)
    # 0.7 = İdare Eder
    # 0.8 = İyi
    # 0.9+ = Mükemmel (Hedefimiz)

    return model, X_test, y_test, preds_prob