import pandas as pd
import numpy as np
import gc  # RAM temizliği (Çöpçü)

# Pandas ayarları: Tabloyu yazdırdığımızda sütunları gizlemesin, hepsini görelim.
pd.set_option('display.max_columns', 500)

print("Veriler yükleniyor... (Bu işlem biraz sürebilir, bekle)")

# 1. ADIM: Verileri diskten RAM'e yüklüyoruz
# input klasörünün içindeki csv dosyalarını okuyoruz.
train_transaction = pd.read_csv('input/train_transaction.csv')
train_identity = pd.read_csv('input/train_identity.csv')

print("Yükleme tamamlandı!")
print(f"Transaction (İşlem) Tablosu Boyutu: {train_transaction.shape}")
print(f"Identity (Kimlik) Tablosu Boyutu: {train_identity.shape}")

# 2. ADIM: Tabloları Birleştirme (Merging)
# SQL Mantığı: Left Join yapıyoruz.
# Amacımız: İşlem tablosunu koruyup, varsa kimlik bilgilerini yanına eklemek.
print("Tablolar birleştiriliyor...")

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

print(f"BİRLEŞMİŞ TABLO BOYUTU (Bunu not et): {train.shape}")

# 3. ADIM: Temizlik (Memory Management)
# Artık birleşmiş 'train' tablosu var, parça parça olanlara ihtiyacımız kalmadı.
# Onları siliyoruz ki bilgisayarın hafızası (RAM) şişmesin.
del train_transaction, train_identity
gc.collect()  # Çöpçüyü çağırıyoruz

print("Gereksiz dosyalar silindi, RAM temizlendi. Analize hazırız!")

# İlk 5 satırı görelim ki neye benziyormuş verimiz
print(train.head())

print("\n" + "#"*30)
print("ADIM 4: HEDEF DEĞİŞKEN ANALİZİ (EDA)")
print("#"*30)

# isFraud sütunu: 0 -> Normal İşlem, 1 -> Dolandırıcılık
# value_counts() fonksiyonu kaç tane 0 kaç tane 1 var sayar.
fraud_counts = train['isFraud'].value_counts()
print("Sayılar:\n", fraud_counts)

# Oransal olarak bakalım (mean bize ortalamayı, yani 1'lerin oranını verir)
fraud_rate = train['isFraud'].mean() * 100
print(f"\nDolandırıcılık Oranı: %{fraud_rate:.2f}")

print("#"*30)