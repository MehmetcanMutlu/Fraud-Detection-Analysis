import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_fraud_analysis(df, output_dir='outputs'):
    """
    Fraud dağılımını ve işlem tutarlarını görselleştirip kaydeder.
    """
    # Eğer outputs klasörü yoksa oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("[INFO] Görselleştirme işlemi başladı...")

    # 1. Pasta Grafiği
    plt.figure(figsize=(8, 8))
    df['isFraud'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], explode=(0, 0.1))
    plt.title('İşlem Dağılımı: Normal vs Fraud')
    plt.ylabel('')
    plt.savefig(f'{output_dir}/1_fraud_distribution_modular.png')
    
    # 2. Histogram (Tutar Analizi)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='TransactionAmt', hue='isFraud', bins=50, log_scale=True, common_norm=False, palette={0: "blue", 1: "red"})
    plt.title('İşlem Tutarı Dağılımı (Logaritmik)')
    plt.xlabel('İşlem Tutarı ($)')
    plt.savefig(f'{output_dir}/2_transaction_amount_modular.png')

    print(f"[INFO] Grafikler '{output_dir}' klasörüne kaydedildi.")