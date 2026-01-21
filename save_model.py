from src.data_loader import load_datasets
from src.preprocessing import merge_data
from src.feature_engineering import run_feature_engineering
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import joblib  # Modeli kaydetmek için kütüphane
import gc

def main():
    print("💾 MODEL KAYDETME OPERASYONU BAŞLADI...")
    
    # 1. Veri Hazırlığı (Aynı süreç)
    train_trans, train_id = load_datasets('input/train_transaction.csv', 'input/train_identity.csv')
    test_trans, test_id = load_datasets('input/test_transaction.csv', 'input/test_identity.csv')
    
    train_df = merge_data(train_trans, train_id)
    test_df = merge_data(test_trans, test_id)
    
    del train_trans, train_id, test_trans, test_id
    gc.collect()
    
    train_len = len(train_df)
    test_df['isFraud'] = -1 
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    print("🧠 Feature Engineering yapılıyor...")
    full_df = run_feature_engineering(full_df)
    
    train_df = full_df.iloc[:train_len]
    test_df = full_df.iloc[train_len:] # Test verisini de demo için kullanacağız
    
    # Test verisinden ID'leri alıp demo için saklayacağız
    demo_data = test_df.drop(['isFraud'], axis=1).sample(100) # Rastgele 100 işlem al
    demo_ids = demo_data['TransactionID'].values
    demo_features = demo_data.drop(['TransactionID'], axis=1)

    y = train_df['isFraud'].astype(int)
    X = train_df.drop(['isFraud', 'TransactionID'], axis=1)
    
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)

    # 2. Modelleri Eğit
    print("🔥 XGBoost Eğitiliyor...")
    model_xgb = xgb.XGBClassifier(
        n_estimators=165, max_depth=12, learning_rate=0.19,
        subsample=0.87, colsample_bytree=0.99, scale_pos_weight=ratio,
        random_state=42, tree_method='hist', eval_metric='auc'
    )
    model_xgb.fit(X, y)

    print("🔥 LightGBM Eğitiliyor...")
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=256,
        scale_pos_weight=ratio, random_state=42, n_jobs=-1, verbose=-1
    )
    model_lgb.fit(X, y)

    # 3. KAYDETME AŞAMASI (Turşu Kuruyoruz 🥒)
    print("📦 Modeller ve Demo verisi paketleniyor...")
    joblib.dump(model_xgb, 'model_xgb.pkl')
    joblib.dump(model_lgb, 'model_lgb.pkl')
    joblib.dump(demo_features, 'demo_data.pkl') # Web sitesinde test etmek için veri
    
    print("✅ BAŞARILI! 'model_xgb.pkl', 'model_lgb.pkl' ve 'demo_data.pkl' oluşturuldu.")

if __name__ == "__main__":
    main()