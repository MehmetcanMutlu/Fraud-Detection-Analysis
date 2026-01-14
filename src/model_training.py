import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

def train_model(df):
    """
    XGBoost modelini eğitir ve test eder.
    """
    print("🤖 Model Eğitimi Başlıyor (XGBoost)...")

    # 1. Hedef ve Özellikleri Ayır
    y = df['isFraud']
    X = df.drop(['isFraud', 'TransactionID'], axis=1)
    
    # 2. Train / Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Eğitim Seti: {X_train.shape}, Test Seti: {X_test.shape}")

    # 3. Dengesizlik Ayarı (Class Imbalance)
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    print(f"[INFO] Dengesizlik Oranı (scale_pos_weight): {ratio:.2f}")

    # 4. Modeli Kur (Gereksiz parametre temizlendi)
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=10,
        learning_rate=0.1,
        scale_pos_weight=ratio,
        eval_metric='auc',
        random_state=42
    )

    # 5. Modeli Eğit
    model.fit(X_train, y_train)
    
    # 6. Tahmin Yap
    preds_prob = model.predict_proba(X_test)[:, 1]
    
    # 7. Performansı Ölç
    auc_score = roc_auc_score(y_test, preds_prob)
    print(f"\n🏆 TEST SONUCU (ROC-AUC Skoru): %{auc_score * 100:.2f}")

    return model, X_test, y_test, preds_prob