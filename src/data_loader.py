import pandas as pd

def load_datasets(transaction_path, identity_path):
    """
    Veri setlerini belirtilen yollardan yükler.
    """
    print(f"[INFO] Veriler yükleniyor: {transaction_path} ve {identity_path}")
    
    # Verileri oku
    train_transaction = pd.read_csv(transaction_path)
    train_identity = pd.read_csv(identity_path)
    
    print("[INFO] Yükleme tamamlandı.")
    return train_transaction, train_identity