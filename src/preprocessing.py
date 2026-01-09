import pandas as pd
import gc

def merge_data(df_trans, df_id):
    """
    Transaction ve Identity tablolarını TransactionID üzerinden birleştirir.
    RAM optimizasyonu yapar.
    """
    print("[INFO] Tablolar birleştiriliyor (Left Join)...")
    
    # Left Join işlemi
    df_merged = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    
    print(f"[INFO] Birleşmiş Tablo Boyutu: {df_merged.shape}")
    
    # RAM Temizliği
    del df_trans, df_id
    gc.collect()
    print("[INFO] Gereksiz değişkenler silindi, RAM temizlendi.")
    
    return df_merged