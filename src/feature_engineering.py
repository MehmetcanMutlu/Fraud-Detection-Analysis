import pandas as pd
import numpy as np

def run_feature_engineering(df):
    """
    Bu fonksiyon ham veriden AKILLI ÖZELLİKLER (Smart Features) türetir.
    Sadece sayıya çevirmekle kalmaz, verinin hikayesini ortaya çıkarır.
    """
    print("🧠 Feature Engineering: Veriye zeka katılıyor...")

    # --- DERS 2: E-POSTA DÜZENLEME (Email Mapping) ---
    # Amacımız: 'yahoo.co.jp' ile 'yahoo.com'u aynı kefeye koymak.
    # E-posta sütunları varsa işlemi yap (bazen sütunlar olmayabilir diye kontrol ediyoruz)
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            # Noktadan (.) böl ve ilk parçayı al (yahoo.co.jp -> yahoo)
            df[col] = df[col].astype(str).apply(lambda x: x.split('.')[0])
            print(f"   -> {col} firmalara göre gruplandı (Google, Yahoo vb.)")

    # --- DERS 3: FREKANS (Count Encoding) ---
    # Amacımız: Bir kart veya adres ne kadar sık kullanılmış? Bot olabilir mi?
    # Bu sütunların her biri için "Kaç kere geçiyor?" sütunu oluşturacağız.
    count_cols = ['card1', 'card2', 'addr1', 'P_emaildomain']
    
    for col in count_cols:
        if col in df.columns:
            # value_counts() sayar, map() ise bu sayıları tabloya yerleştirir
            df[f'{col}_count'] = df[col].map(df[col].value_counts(dropna=False))
            print(f"   -> {col}_count oluşturuldu (Sıklık analizi)")

    # --- DERS 1: GRUPLAMA VE ORANLAR (Aggregations) ---
    # Amacımız: Harcama tutarı (TransactionAmt) normal mi yoksa ortalamadan sapmış mı?
    # card1 (Kart Tipi) bazında ortalamayı alıyoruz.
    if 'card1' in df.columns and 'TransactionAmt' in df.columns:
        # 1. Kartın ortalama harcaması nedir?
        df['TransactionAmt_mean_card1'] = df.groupby('card1')['TransactionAmt'].transform('mean')
        
        # 2. Kartın harcama standart sapması (oynaklığı) nedir?
        df['TransactionAmt_std_card1'] = df.groupby('card1')['TransactionAmt'].transform('std')
        
        # 3. ŞİMDİKİ HARCAMA / ORTALAMA HARCAMA (En Kritik Özellik!)
        # Eğer bu sayı 10 ise, kişi normalden 10 kat fazla harcamış demektir.
        df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df['TransactionAmt_mean_card1']
        
        print("   -> TransactionAmt analizleri yapıldı (Ortalamadan sapma hesaplandı)")

    # --- TEMİZLİK: LABEL ENCODING (Eski Kodumuz) ---
    # Model sadece sayı anlar. Kalan tüm yazıları (String) sayıya çeviriyoruz.
    print("🧮 Yazılar sayıya çevriliyor (Label Encoding)...")
    for col in df.columns:
        if df[col].dtype == 'object': # Eğer sütun yazı ise
            # Kategorik tipe çevir ve kodla (0, 1, 2...)
            df[col] = df[col].astype('category').cat.codes

    # Sonsuz sayıları (bölme işleminden çıkan) temizle
    df = df.replace([np.inf, -np.inf], np.nan)
    # Boşlukları (NaN) -999 ile doldur (XGBoost bunu anlar)
    df = df.fillna(-999)

    print("✅ Feature Engineering tamamlandı!")
    return df