import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def plot_fraud_analysis(df, output_dir='outputs'):
    """
    Fraud dağılımını ve işlem tutarlarını görselleştirip kaydeder.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("[INFO] Görselleştirme işlemi başladı...")

    # 1. Pasta Grafiği
    plt.figure(figsize=(8, 8))
    df['isFraud'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], explode=(0, 0.1))
    plt.title('İşlem Dağılımı: Normal vs Fraud')
    plt.ylabel('')
    plt.savefig(f'{output_dir}/fraud_distribution.png')
    
    # 2. Histogram (Tutar Analizi)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='TransactionAmt', hue='isFraud', bins=50, log_scale=True, common_norm=False, palette={0: "blue", 1: "red"})
    plt.title('İşlem Tutarı Dağılımı (Logaritmik)')
    plt.xlabel('İşlem Tutarı ($)')
    plt.savefig(f'{output_dir}/transaction_amount.png')

    print(f"[INFO] Temel analiz grafikleri '{output_dir}' klasörüne kaydedildi.")

def plot_feature_importance(model, feature_names, output_dir='outputs'):
    """
    Modelin en çok önem verdiği özellikleri sıralar ve çizer.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("[INFO] Özellik Önem Düzeyi (Feature Importance) çiziliyor...")
    
    # Özellik önemlerini al
    importance = model.feature_importances_
    
    # DataFrame'e çevirip sırala
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False).head(20) # En önemli 20 özellik
    
    # Görselleştir
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Model İçin En Kritik 20 Özellik (Feature Importance)')
    plt.xlabel('Önem Düzeyi')
    plt.ylabel('Özellik Adı')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png')
    
    print(f"[INFO] Kritik özellikler grafiği kaydedildi: {output_dir}/feature_importance.png")