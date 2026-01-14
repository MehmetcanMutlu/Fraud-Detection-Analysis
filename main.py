from src.data_loader import load_datasets
from src.preprocessing import merge_data
from src.feature_engineering import run_feature_engineering
from src.model_training import train_model
from src.visualization import plot_fraud_analysis, plot_feature_importance
import pandas as pd

# Pandas ayarları
pd.set_option('display.max_columns', 500)

def main():
    print("🚀 PROJE BAŞLATILIYOR")
    print("="*40)
    
    # 1. Adım: Veri Yükleme
    df_trans, df_id = load_datasets(
        transaction_path='input/train_transaction.csv',
        identity_path='input/train_identity.csv'
    )
    
    # 2. Adım: Birleştirme
    train_df = merge_data(df_trans, df_id)
    
    # 3. Adım: Feature Engineering
    train_df = run_feature_engineering(train_df)
    
    # 4. Adım: Temel Analiz Görselleştirmesi
    # (Her seferinde çalışmasına gerek yoksa başındaki # işaretini kaldırıp yorum satırı yapabilirsin)
    # plot_fraud_analysis(train_df) 

    # 5. Adım: Model Eğitimi
    model, X_test, y_test, preds = train_model(train_df)
    
    # 6. Adım: Özellik Önem Analizi (YENİ)
    # Modelin hangi sütunlara (feature) dikkat ettiğini çiziyoruz
    # Not: X_test.columns diyerek özellik isimlerini veriyoruz
    plot_feature_importance(model, X_test.columns)
    
    print("="*40)
    print("✅ PROJE TAMAMLANDI: Model eğitildi ve analizler 'outputs' klasörüne kaydedildi.")

if __name__ == "__main__":
    main() 