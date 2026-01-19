from src.data_loader import load_datasets
from src.preprocessing import merge_data
from src.feature_engineering import run_feature_engineering
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import gc

def main():
    print("🚀 ENSEMBLE MODU BAŞLATILIYOR (XGBoost + LightGBM)")
    print("="*40)

    # 1. VERİ YÜKLEME
    print("[1/6] Veriler yükleniyor...")
    try:
        train_trans, train_id = load_datasets('input/train_transaction.csv', 'input/train_identity.csv')
        test_trans, test_id = load_datasets('input/test_transaction.csv', 'input/test_identity.csv')
    except FileNotFoundError:
        print("❌ HATA: Dosyalar yok!")
        return

    # 2. MERGE
    print("\n[2/6] Birleştiriliyor...")
    train_df = merge_data(train_trans, train_id)
    test_df = merge_data(test_trans, test_id)
    
    del train_trans, train_id, test_trans, test_id
    gc.collect()

    train_len = len(train_df)
    test_df['isFraud'] = -1 
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    del train_df, test_df
    gc.collect()

    # 3. FEATURE ENGINEERING
    print("\n[3/6] Feature Engineering çalıştırılıyor...")
    full_df = run_feature_engineering(full_df)

    # 4. AYIRMA
    print("\n[4/6] Veriler ayrılıyor...")
    train_df = full_df.iloc[:train_len]
    test_df = full_df.iloc[train_len:]
    
    test_ids = test_df['TransactionID']
    test_df = test_df.drop(['isFraud', 'TransactionID'], axis=1)
    
    y = train_df['isFraud'].astype(int)
    X = train_df.drop(['isFraud', 'TransactionID'], axis=1)
    
    del full_df
    gc.collect()

    ratio = float(np.sum(y == 0)) / np.sum(y == 1)

    # --- MODEL 1: XGBOOST ---
    print("\n[5/6] MODELLER EĞİTİLİYOR...")
    print("   -> Model 1: XGBoost eğitiliyor (Optuna Ayarlı)...")
    
    clf_xgb = xgb.XGBClassifier(
        n_estimators=165,
        max_depth=12,
        learning_rate=0.190197,
        subsample=0.86821,
        colsample_bytree=0.98775,
        scale_pos_weight=ratio,
        random_state=42,
        tree_method='hist',
        eval_metric='auc'
    )
    clf_xgb.fit(X, y, verbose=True)
    pred_xgb = clf_xgb.predict_proba(test_df)[:, 1]
    print("   ✅ XGBoost tamamlandı.")

    # --- MODEL 2: LIGHTGBM ---
    print("   -> Model 2: LightGBM eğitiliyor (Hızlı ve Öfkeli)...")
    
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=256,
        scale_pos_weight=ratio,
        random_state=42,
        n_jobs=-1,
        verbose=-1 # Gereksiz uyarıları gizle
    )
    clf_lgb.fit(X, y)
    pred_lgb = clf_lgb.predict_proba(test_df)[:, 1]
    print("   ✅ LightGBM tamamlandı.")

    # --- ENSEMBLE ---
    print("\n[6/6] Güçler birleştiriliyor (%50 - %50)...")
    
    final_preds = (0.5 * pred_xgb) + (0.5 * pred_lgb)
    
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': final_preds
    })
    
    submission.to_csv('submission_ensemble.csv', index=False)
    print("\n🎉 TEBRİKLER! 'submission_ensemble.csv' hazır.")

if __name__ == "__main__":
    main()