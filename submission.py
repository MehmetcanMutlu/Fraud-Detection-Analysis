from src.data_loader import load_datasets
from src.preprocessing import merge_data
from src.feature_engineering import run_feature_engineering
import pandas as pd
import xgboost as xgb
import numpy as np
import gc

def main():
    print("🚀 KAGGLE SUBMISSION MODU BAŞLATILIYOR")
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

    # ⚠️ KRİTİK NOKTA (HATAYI ÇÖZEN KISIM)
    # Etiket (dataset_type) kullanmak yerine, train setinin uzunluğunu kaydediyoruz.
    # Feature Engineering sonrası bu sayıdan kesip ayıracağız.
    train_len = len(train_df)
    print(f"[BILGI] Train seti uzunluğu kaydedildi: {train_len}")

    # Test setinde 'isFraud' sütunu yok, hata vermesin diye geçici olarak ekliyoruz.
    test_df['isFraud'] = -1 

    # 3. TEK PARÇA HALİNE GETİR (Concat)
    print("\n3. Adım: Feature Engineering için birleştiriliyor...")
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    del train_df, test_df
    gc.collect()

    # 4. FEATURE ENGINEERING
    full_df = run_feature_engineering(full_df)

    # 5. TEKRAR AYIR (Index Slicing ile)
    print("\n4. Adım: Veriler tekrar ayrılıyor (Index Slicing)...")
    
    # 0'dan train_len'e kadar olanlar TRAIN
    train_df = full_df.iloc[:train_len]
    
    # train_len'den sonuna kadar olanlar TEST
    test_df = full_df.iloc[train_len:]
    
    # Test verisinden geçici isFraud sütununu atalım
    test_ids = test_df['TransactionID'] # ID'leri sakla
    test_df = test_df.drop(['isFraud', 'TransactionID'], axis=1) # Temizle
    
    del full_df
    gc.collect()

    # 6. MODEL EĞİTİMİ
    print("\n5. Adım: Model eğitiliyor (Full Train Seti ile)...")
    
    y = train_df['isFraud'].astype(int)
    X = train_df.drop(['isFraud', 'TransactionID'], axis=1)
    
    # Dengesizlik ayarı
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    print(f"[BILGI] Pos/Neg Oranı: {ratio:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        random_state=42,
        tree_method='hist'
    )
    
    model.fit(X, y, verbose=True)
    print("✅ Model eğitimi tamamlandı.")

    # 7. TAHMİN VE KAYIT
    print("\n6. Adım: Submission dosyası hazırlanıyor...")
    
    preds = model.predict_proba(test_df)[:, 1]
    
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("\n🎉 TEBRİKLER! 'submission.csv' dosyası oluşturuldu.")
    print("GitHub'a atmadan önce Kaggle'a yükleyip sıranı görebilirsin!")

if __name__ == "__main__":
    main()