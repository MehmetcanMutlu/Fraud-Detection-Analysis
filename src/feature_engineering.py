import pandas as pd
from sklearn.preprocessing import LabelEncoder

def identify_columns(df):
    """
    Hangi sütunlar sayısal (numerical), hangileri kategorik (categorical) belirler.
    """
    # Object (String) olanlar kategoriktir
    cat_cols = list(df.select_dtypes(include=['object']).columns)
    
    # Geri kalanlar sayısaldır (isFraud hariç)
    num_cols = [c for c in df.columns if c not in cat_cols and c != 'isFraud' and c != 'TransactionID']
    
    print(f"[INFO] Kategorik Sütun Sayısı: {len(cat_cols)}")
    print(f"[INFO] Sayısal Sütun Sayısı: {len(num_cols)}")
    
    return cat_cols, num_cols

def handle_missing_values(df, cat_cols, num_cols):
    """
    Boş değerleri (NaN) doldurur.
    Strateji:
    - Kategorikler: 'Unknown' olarak doldurulur.
    - Sayısallar: -999 ile doldurulur (Tree modelleri için bu 'boş' anlamına gelir).
    """
    print("[INFO] Eksik veriler dolduruluyor...")
    
    # Kategorik boşlukları 'Unknown' yap
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        
    # Sayısal boşlukları -999 yap
    for col in num_cols:
        df[col] = df[col].fillna(-999)
        
    return df

def encode_categoricals(df, cat_cols):
    """
    Yazı olan kategorik verileri sayıya çevirir (Label Encoding).
    Örn: 'Gmail' -> 1, 'Yahoo' -> 2
    """
    print("[INFO] Label Encoding yapılıyor...")
    
    for col in cat_cols:
        le = LabelEncoder()
        # Veriyi string'e çevirip fit ediyoruz (garanti olsun diye)
        df[col] = le.fit_transform(df[col].astype(str))
        
    print("[INFO] Encoding tamamlandı. Artık tüm veri sayısal.")
    return df

def run_feature_engineering(df):
    """
    Tüm işlemleri sırasıyla çalıştıran ana fonksiyon.
    """
    print("\n🚀 Feature Engineering Başladı...")
    
    # 1. Sütun tiplerini bul
    cat_cols, num_cols = identify_columns(df)
    
    # 2. Boşlukları doldur
    df = handle_missing_values(df, cat_cols, num_cols)
    
    # 3. Yazıları sayıya çevir
    df = encode_categoricals(df, cat_cols)
    
    print("✅ Feature Engineering Tamamlandı.\n")
    return df