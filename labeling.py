import os
import time
import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import APIError

# 1. Inisialisasi API Client Gemini
# Ganti dengan API Key kamu yang baru dari Google AI Studio
API_KEY = "xxx-xxx-xxx"  # <-- Ganti dengan API Key kamu
client = genai.Client(api_key=API_KEY)

def analisis_sentimen(kalimat, retries=3, delay=4):
    """
    Fungsi untuk mengirim teks ke Gemini API dan mengembalikan label sentimen.
    Dilengkapi dengan fitur auto-retry jika server mengalami high demand (503).
    """
    # Jika baris kosong atau bukan string, langsung beri label NETRAL atau kosong
    if not isinstance(kalimat, str) or not kalimat.strip():
        return "NETRAL"

    prompt = f"""
    Analisis sentimen dari kalimat berikut (bisa berupa teks/komentar media sosial).
    Klasifikasikan menjadi salah satu dari kategori ini: POSITIF, NEGATIF, atau NETRAL.
    Berikan jawaban HANYA satu kata kategori tersebut tanpa penjelasan tambahan.

    Kalimat: "{kalimat}"
    """

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=prompt,
                # Menggunakan temperature rendah agar model konsisten kaku pada pilihan kategori
                config=types.GenerateContentConfig(
                    temperature=0.1
                )
            )

            # Membersihkan output dari spasi atau karakter newline
            hasil = response.text.strip().upper()

            # Validasi agar output benar-to-benar salah satu dari 3 kategori utama
            if "POSITIF" in hasil:
                return "POSITIF"
            elif "NEGATIF" in hasil:
                return "NEGATIF"
            else:
                return "NETRAL"

        except APIError as e:
            # Handle error 503 (High Demand) atau 429 (Rate Limit)
            if e.code in [503, 429] and attempt < retries - 1:
                print(f" -> Server sibuk/limit (Error {e.code}). Mencoba kembali dalam {delay} detik... (Percobaan {attempt + 1}/{retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            else:
                print(f" -> Gagal memproses baris ini karena API Error: {e}")
                return "ERROR_API"
        except Exception as e:
            print(f" -> Error tidak terduga: {e}")
            return "ERROR_UNKNOWN"

def main():
    input_file = "gabungFromKevin.csv"
    output_file = "gabungFromKevin_labeled.csv"

    # 2. Baca file CSV dataset
    print(f"Membaca file {input_file}...")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: File '{input_file}' tidak ditemukan di direktori ini!")
        return

    if 'comment' not in df.columns:
        print("Error: Kolom 'comment' tidak ditemukan di dalam file CSV kamu.")
        return

    # 3. Looping untuk memproses analisis sentimen baris demi baris
    hasil_sentimen = []
    total_baris = len(df)
    print(f"Memulai analisis untuk {total_baris} data...")

    for index, row in df.iterrows():
        komentar = row['comment']
        print(f"Memproses data ke-{index + 1}/{total_baris}...", end="", flush=True)

        # Panggil fungsi analisis
        label = analisis_sentimen(komentar)
        hasil_sentimen.append(label)
        print(f" [Hasil: {label}]")

        # Delay tipis (0.5 detik) antar baris agar tidak membebani rate limit kuota gratis
        time.sleep(0.5)

    # 4. Masukkan list hasil ke dalam kolom baru di DataFrame
    df['sentiment'] = hasil_sentimen

    # 5. Simpan ke file CSV baru
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSelesai! Hasil analisis sukses disimpan ke file: '{output_file}'")

if __name__ == "__main__":
    main()