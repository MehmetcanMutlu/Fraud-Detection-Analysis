import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Bizim modüller
from src.data_loader import load_datasets
from src.preprocessing import merge_data
from src.feature_engineering import run_feature_engineering

def objective(trial, X, y):
    """
    Optuna'nın her denemede (trial) çalıştıracağı fonksiyon.
    Burada rastgele parametreler seçilir ve model test edilir.
    """
    
    # 1. Hiperparametre Arama Alanı (Search Space)
    # Optuna'ya diyoruz ki: "Bu aralıklarda gez"
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # Sabit Parametreler
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist', # Hızlandırma için
        'random_state': 42,
        'n_jobs': -1 # Tüm işlemci çekirdeklerini kullan
    }

    # Dengesizlik ayarını ekle (scale_pos_weight)
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    params['scale_pos_weight'] = ratio

    # 2. Veriyi Böl (Train / Validation)
    # Her denemede %20'lik kısmı ayırıp test ediyoruz
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 3. Modeli Eğit
    model = xgb.XGBClassifier(**params)
    model.fit(train_x, train_y)

    # 4. Skoru Hesapla
    preds = model.predict_proba(valid_x)[:, 1]
    auc = roc_auc_score(valid_y, preds)

    return auc

def run_optimization():
    print("🚀 OPTUNA OPTİMİZASYONU BAŞLIYOR...")
    print("="*40)

    # --- VERİ HAZIRLIĞI (Submission.py ile aynı mantık) ---
    print("[1/4] Veriler yükleniyor...")
    # Sadece Train verisi yeterli, optimizasyonu orada yapacağız
    df_trans, df_id = load_datasets('input/train_transaction.csv', 'input/train_identity.csv')
    df = merge_data(df_trans, df_id)
    
    del df_trans, df_id
    gc.collect()

    print("[2/4] Feature Engineering yapılıyor...")
    df = run_feature_engineering(df)

    # Hedef ve Özellikler
    y = df['isFraud']
    X = df.drop(['isFraud', 'TransactionID'], axis=1)

    del df
    gc.collect()

    # --- OPTUNA ÇALIŞTIRMA ---
    print("\n[3/4] Robot çalışmaya başladı (Bu işlem zaman alabilir)...")
    
    # Study: Optuna'nın çalışma defteri
    # direction='maximize' -> Çünkü AUC skorunun yüksek olmasını istiyoruz
    study = optuna.create_study(direction='maximize')
    
    # lambda fonksiyonu ile veriyi objective fonksiyonuna taşıyoruz
    study.optimize(lambda trial: objective(trial, X, y), n_trials=20) 

    # --- SONUÇLAR ---
    print("\n" + "="*40)
    print("🏆 EN İYİ SONUÇLAR BULUNDU!")
    print(f"En Yüksek AUC Skoru: {study.best_value:.5f}")
    print("En İyi Parametreler:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*40)
    print("👉 Şimdi bu parametreleri 'submission.py' dosyana kopyalayabilirsin!")

if __name__ == "__main__":
    run_optimization()