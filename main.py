# Modüllerimizi çağırıyoruz
from src.data_loader import load_datasets
from src.preprocessing import merge_data
from src.visualization import plot_fraud_analysis
import pandas as pd

# Pandas ayarları
pd.set_option('display.max_columns', 500)

def main():
    print("🚀 PROJE BAŞLATILIYOR (MODÜLER YAPI)")
    print("="*40)
    
    # 1. Adım: Veri Yükleme
    df_trans, df_id = load_datasets(
        transaction_path='input/train_transaction.csv',
        identity_path='input/train_identity.csv'
    )
    
    # 2. Adım: Birleştirme
    train_df = merge_data(df_trans, df_id)
    
    # 3. Adım: Analiz & Görselleştirme
    fraud_rate = train_df['isFraud'].mean() * 100
    print(f"\n📊 DOLANDIRICILIK ORANI: %{fraud_rate:.2f}\n")
    
    plot_fraud_analysis(train_df)
    
    print("="*40)
    print("✅ İŞLEM BAŞARIYLA TAMAMLANDI")

if __name__ == "__main__":
    main()