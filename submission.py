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

    # 1. VERİLERİ YÜKLE (Train + Test)
    # input klasöründe test_transaction.csv ve test_identity.csv olduğundan emin ol!
    print("1. Adım: Train ve Test verileri yükleniyor...")
    try:
        train_trans, train_id = load_datasets('input/train_transaction.csv', 'input/train_identity.csv')
        test_trans, test_id = load_datasets('input/test_transaction.csv', 'input/test_identity.csv')
    except FileNotFoundError:
        print("❌ HATA: Test dosyaları bulunamadı! 'input' klasörüne test_transaction.csv ve test_identity.csv dosyalarını koymalısın.")
        return

    # 2. BİRLEŞTİR (Merge)
    print("\n2. Adım: Tablolar birleştiriliyor...")
    train_df = merge_data(train_trans, train_id)
    test_df = merge_data(test_trans, test_id)
    
    # RAM Temizliği
    del train_trans, train_id, test_trans, test_id
    gc.collect()

    # 3. KARIŞIKLIK OLMASIN DİYE ETİKETLEME
    train_df['dataset_type'] = 'train'
    test_df['dataset_type'] = 'test'
    test_df['isFraud'] = -1 # Test setinde cevaplar yok, geçici değer

    # 4. TEK PARÇA HALİNE GETİR (Concat)
    print("\n3. Adım: Train ve Test setleri birleştiriliyor (Feature Engineering için)...")
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    del train_df, test_df
    gc.collect()

    # 5. FEATURE ENGINEERING (Sayıya Çevirme)
    # Burası biraz uzun sürebilir (1 milyondan fazla satır)
    full_df = run_feature_engineering(full_df)

    # 6. TEKRAR AYIR
    print("\n4. Adım: Veriler tekrar Train/Test olarak ayrılıyor...")
    train_df = full_df[full_df['dataset_type'] == 'train'].drop('dataset_type', axis=1)
    test_df = full_df[full_df['dataset_type'] == 'test'].drop(['dataset_type', 'isFraud'], axis=1)
    
    # ID'leri sakla (Kaggle istiyor)
    test_ids = test_df['TransactionID']
    
    del full_df
    gc.collect()

    # 7. MODEL EĞİTİMİ (Full Train Data ile)
    print("\n5. Adım: Model eğitiliyor (Full Train Seti ile)...")
    
    y = train_df['isFraud'].astype(int)
    X = train_df.drop(['isFraud', 'TransactionID'], axis=1)
    
    # Dengesizlik ayarı
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    
    model = xgb.XGBClassifier(
        n_estimators=500,       # Daha güçlü (500 ağaç)
        max_depth=10,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        random_state=42,
        tree_method='hist'      # Hızlandırma
    )
    
    model.fit(X, y, verbose=True)
    print("✅ Model eğitimi tamamlandı.")

    # 8. TAHMİN VE KAYIT
    print("\n6. Adım: Test seti üzerinde tahmin yapılıyor...")
    X_test = test_df.drop('TransactionID', axis=1)
    
    # Olasılık tahmini al (0 ile 1 arası)
    preds = model.predict_proba(X_test)[:, 1] 
    
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("\n🎉 TEBRİKLER! 'submission.csv' dosyası oluşturuldu.")
    print("Şimdi bu dosyayı Kaggle'a yükleyebilirsin!")

if __name__ == "__main__":
    main()