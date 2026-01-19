from src.data_loader import load_datasets
from src.preprocessing import merge_data
from src.feature_engineering import run_feature_engineering
import pandas as pd
import xgboost as xgb
import numpy as np
import gc

def main():
    print("🚀 KAGGLE SUBMISSION MODU BAŞLATILIYOR (OPTUNA AYARLARI)")
    print("="*40)

    # 1. VERİLERİ YÜKLE
    print("1. Adım: Train ve Test verileri yükleniyor...")
    try:
        train_trans, train_id = load_datasets('input/train_transaction.csv', 'input/train_identity.csv')
        test_trans, test_id = load_datasets('input/test_transaction.csv', 'input/test_identity.csv')
    except FileNotFoundError:
        print("❌ HATA: Dosyalar bulunamadı! Input klasörünü kontrol et.")
        return

    # 2. BİRLEŞTİR (Merge)
    print("\n2. Adım: Tablolar birleştiriliyor...")
    train_df = merge_data(train_trans, train_id)
    test_df = merge_data(test_trans, test_id)
    
    # Hafıza temizliği
    del train_trans, train_id, test_trans, test_id
    gc.collect()

    # Data Leakage Önlemi: Index Slicing için uzunluğu kaydet
    train_len = len(train_df)
    print(f"[BILGI] Train seti uzunluğu kaydedildi: {train_len}")

    # Test setinde 'isFraud' sütunu yok, hata vermesin diye geçici ekliyoruz
    test_df['isFraud'] = -1 

    # 3. TEK PARÇA HALİNE GETİR (Concat)
    print("\n3. Adım: Feature Engineering için birleştiriliyor...")
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    del train_df, test_df
    gc.collect()

    # 4. FEATURE ENGINEERING
    # Tüm veriyi (Train + Test) aynı anda işliyoruz ki tutarlı olsun
    full_df = run_feature_engineering(full_df)

    # 5. TEKRAR AYIR (Index Slicing)
    print("\n4. Adım: Veriler tekrar ayrılıyor (Index Slicing)...")
    
    # 0'dan train_len'e kadar olanlar TRAIN
    train_df = full_df.iloc[:train_len]
    
    # train_len'den sonuna kadar olanlar TEST
    test_df = full_df.iloc[train_len:]
    
    # Test verisinden geçici isFraud ve ID'leri temizle
    test_ids = test_df['TransactionID']
    test_df = test_df.drop(['isFraud', 'TransactionID'], axis=1)
    
    del full_df
    gc.collect()

    # 6. MODEL EĞİTİMİ (SÜPER AYARLAR İLE)
    print("\n5. Adım: Model eğitiliyor (Optuna ile optimize edildi)...")
    
    y = train_df['isFraud'].astype(int)
    X = train_df.drop(['isFraud', 'TransactionID'], axis=1)
    
    # Dengesizlik ayarı
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    print(f"[BILGI] Pos/Neg Oranı (scale_pos_weight): {ratio:.2f}")
    
    # 🔥 OPTUNA ROBOTUNUN BULDUĞU PARAMETRELER 🔥
    model = xgb.XGBClassifier(
        n_estimators=165,           # Robot buldu
        max_depth=12,               # Robot buldu (Derin öğrenme)
        learning_rate=0.190197,     # Robot buldu
        subsample=0.86821,          # Robot buldu
        colsample_bytree=0.98775,   # Robot buldu
        scale_pos_weight=ratio,     # Dengesizlik ayarı (Sabit)
        random_state=42,
        tree_method='hist',         # Hızlandırma
        eval_metric='auc'
    )
    
    model.fit(X, y, verbose=True)
    print("✅ Model eğitimi tamamlandı.")

    # 7. TAHMİN VE KAYIT
    print("\n6. Adım: Submission dosyası hazırlanıyor...")
    
    # Olasılık tahmini (0-1 arası)
    preds = model.predict_proba(test_df)[:, 1]
    
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("\n🎉 TEBRİKLER! 'submission.csv' dosyası oluşturuldu.")
    print("🚀 Kaggle'a yüklemeye hazırsın! (Hedef: 0.92+ Public Score)")

if __name__ == "__main__":
    main()