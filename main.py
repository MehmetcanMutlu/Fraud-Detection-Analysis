from src.data_loader import load_datasets
from src.preprocessing import merge_data
from src.feature_engineering import run_feature_engineering
from src.model_training import train_model  # <-- YENİ EKLEDİK
from src.visualization import plot_fraud_analysis
import pandas as pd

# Pandas ayarları
pd.set_option('display.max_columns', 500)

def main():
    print("🚀 PROJE BAŞLATILIYOR")
    print("="*40)
    
    # 1. Yükle
    df_trans, df_id = load_datasets(
        transaction_path='input/train_transaction.csv',
        identity_path='input/train_identity.csv'
    )
    
    # 2. Birleştir
    train_df = merge_data(df_trans, df_id)
    
    # 3. Özellik Mühendisliği (Sayıya Çevir)
    train_df = run_feature_engineering(train_df)
    
    # 4. Görselleştirme (İsteğe bağlı, tekrar tekrar çizmesin diye yorum satırı yapabilirsin)
    # plot_fraud_analysis(train_df) 

    # 5. MODEL EĞİTİMİ (YENİ)
    # Modeli alıyoruz, tahminleri alıyoruz
    model, X_test, y_test, preds = train_model(train_df)
    
    print("="*40)
    print("✅ PROJE TAMAMLANDI: Model başarıyla eğitildi.")

if __name__ == "__main__":
    main()