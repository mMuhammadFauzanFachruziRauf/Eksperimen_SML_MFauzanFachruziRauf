import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import os

# --- Konfigurasi ---
# Variabel global untuk konfigurasi jika diperlukan
COIN_ID = "bitcoin"
VS_CURRENCY = "idr"
DAYS = "365" # Sesuai dengan limitasi API gratis CoinGecko
OUTPUT_DIR = "namadataset_preprocessing" # Nama folder output di dalam 'preprocessing'
OUTPUT_FILENAME = "bitcoin_idr_daily_processed.csv"

# Pastikan direktori output ada
# Path relatif dari lokasi skrip automate_Nama-Anda.py
# Skrip ini ada di 'preprocessing/', jadi '../preprocessing/namadataset_preprocessing/'
# atau bisa juga langsung 'namadataset_preprocessing/' jika kita menganggap current working directory
# adalah tempat skrip dijalankan. Untuk lebih aman, kita buat path absolut dari skrip.
# Namun, untuk kesederhanaan dan sesuai struktur yang diminta, kita asumsikan
# skrip dijalankan dari root folder proyek atau folder 'preprocessing'.
# Kriteria meminta struktur: preprocessing/namadataset_preprocessing
# Jadi, jika skrip ada di preprocessing/, maka pathnya cukup OUTPUT_DIR.

# Membuat path ke folder output. Diasumsikan skrip ini ada di folder 'preprocessing'
# dan folder 'namadataset_preprocessing' juga ada di dalam 'preprocessing'.
# Jika dijalankan dari root proyek: 'preprocessing/namadataset_preprocessing'
# Jika dijalankan dari folder 'preprocessing': 'namadataset_preprocessing'

# Kita akan buat path relatif terhadap lokasi skrip ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR) # Path ke preprocessing/namadataset_preprocessing

if not os.path.exists(PREPROCESSING_DIR):
    os.makedirs(PREPROCESSING_DIR)
    print(f"Folder '{PREPROCESSING_DIR}' telah dibuat.")

OUTPUT_FILEPATH = os.path.join(PREPROCESSING_DIR, OUTPUT_FILENAME)


def fetch_bitcoin_data(coin_id, vs_currency, days):
    """
    Mengambil data harga Bitcoin dari API CoinGecko.
    """
    print(f"Mengambil data untuk {coin_id} vs {vs_currency} untuk {days} hari terakhir...")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}&interval=daily"
    response = requests.get(url)

    if response.status_code == 200:
        data_json = response.json()
        prices_data = data_json.get('prices')
        if prices_data:
            df_raw = pd.DataFrame(prices_data, columns=['timestamp', 'price'])
            df_raw['date'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
            df_raw = df_raw.set_index('date')
            df_raw = df_raw.drop('timestamp', axis=1)
            print("Data mentah berhasil diambil dari CoinGecko.")
            return df_raw
        else:
            print("Error: Key 'prices' tidak ditemukan dalam respons JSON.")
            return pd.DataFrame()
    else:
        print(f"Error: Gagal mengambil data. Status code: {response.status_code}")
        print(f"Respons API: {response.text}")
        return pd.DataFrame()

def preprocess_data(df_raw):
    """
    Melakukan preprocessing data: feature engineering dan normalisasi.
    """
    if df_raw.empty:
        print("DataFrame mentah kosong, preprocessing dibatalkan.")
        return pd.DataFrame(), None # Kembalikan juga None untuk scaler

    print("Memulai preprocessing data...")
    df_processed = df_raw.copy()

    # 1. Penanganan Missing Values (jika ada - seharusnya tidak untuk kasus ini)
    initial_nans = df_processed.isnull().sum().sum()
    if initial_nans > 0:
        df_processed.fillna(method='ffill', inplace=True)
        df_processed.fillna(method='bfill', inplace=True)
        print(f"{initial_nans} missing values ditangani.")
    else:
        print("Tidak ada missing values awal untuk ditangani.")

    # 2. Feature Engineering
    lags = [1, 3, 7]
    for lag in lags:
        df_processed[f'price_lag_{lag}'] = df_processed['price'].shift(lag)
    print(f"Lag features ditambahkan untuk {lags} hari.")

    rolling_window_size = 7
    df_processed[f'price_rolling_mean_{rolling_window_size}'] = df_processed['price'].rolling(window=rolling_window_size).mean()
    print(f"Rolling mean feature (window {rolling_window_size}) ditambahkan.")

    # Hapus baris dengan NaN yang muncul setelah feature engineering
    rows_before_dropna = len(df_processed)
    df_processed.dropna(inplace=True)
    rows_after_dropna = len(df_processed)
    print(f"{rows_before_dropna - rows_after_dropna} baris dihapus karena NaN dari feature engineering.")

    if df_processed.empty:
        print("DataFrame menjadi kosong setelah dropna, mungkin terlalu sedikit data awal atau window terlalu besar.")
        return pd.DataFrame(), None

    # 3. Normalisasi/Standarisasi Fitur
    scaler = MinMaxScaler()
    features_to_scale = df_processed.columns
    df_processed[features_to_scale] = scaler.fit_transform(df_processed[features_to_scale])
    print("Data telah dinormalisasi menggunakan MinMaxScaler.")

    print("Preprocessing data selesai.")
    return df_processed, scaler # Kembalikan juga scaler jika perlu disimpan atau digunakan nanti

def main():
    """
    Fungsi utama untuk menjalankan seluruh alur kerja preprocessing.
    """
    print("--- Memulai Proses Otomatisasi Preprocessing ---")

    # Langkah 1: Ambil data
    df_raw = fetch_bitcoin_data(COIN_ID, VS_CURRENCY, DAYS)

    if not df_raw.empty:
        # Langkah 2: Preprocess data
        df_final, scaler_obj = preprocess_data(df_raw) # Tangkap scaler jika dikembalikan

        if not df_final.empty:
            # Langkah 3: Simpan data yang sudah diproses
            try:
                df_final.to_csv(OUTPUT_FILEPATH)
                print(f"Data yang sudah diproses berhasil disimpan di: {OUTPUT_FILEPATH}")
                print(f"Jumlah baris: {len(df_final)}, Jumlah kolom: {len(df_final.columns)}")
                print("\nBeberapa baris data yang disimpan:")
                print(df_final.head())
            except Exception as e:
                print(f"Error saat menyimpan file CSV: {e}")
        else:
            print("Tidak ada data untuk disimpan karena proses preprocessing menghasilkan DataFrame kosong.")
    else:
        print("Tidak ada data untuk diproses lebih lanjut.")

    print("--- Proses Otomatisasi Preprocessing Selesai ---")

if __name__ == "__main__":
    main()