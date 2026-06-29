# 📘 FAKTA — Panduan Lengkap Dari Awal Sampai Berjalan

Panduan step-by-step untuk membangun sistem fact-checking AI FAKTA dari nol.

---

## 📋 Daftar Isi

1. [Persiapan Environment](#1-persiapan-environment)
2. [Setup API Key Gemini](#2-setup-api-key-gemini)
3. [Kumpulkan Dataset](#3-kumpulkan-dataset)
4. [Train Model LSTM](#4-train-model-lstm) — **Lokal atau Google Colab**
5. [Setup Evidence Retrieval](#5-setup-evidence-retrieval)
6. [Jalankan Sistem](#6-jalankan-sistem)
7. [Testing & Evaluasi](#7-testing--evaluasi)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Persiapan Environment

### 1.1 Pastikan Python Terinstall

Cek versi Python:
```bash
python --version
```

Harus **Python 3.10+**. Kalau belum, download dari https://www.python.org/downloads/

### 1.2 Buka Terminal, Masuk ke Folder FAKTA

```bash
cd "C:\Users\juwon\onedrive\documents\kuliah ppti25\cawu 3\AI\FAKTA"
```

### 1.3 Buat Virtual Environment

Virtual environment mengisolasi dependencies project ini dari project lain.

```bash
python -m venv venv
```

### 1.4 Activate Virtual Environment

**Windows (CMD):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

Kalau berhasil, prompt akan jadi seperti ini:
```
(venv) C:\Users\juwon\...\FAKTA>
```

### 1.5 Install Semua Dependencies

```bash
pip install -r requirements.txt
```

Ini akan install ~21 package. Proses ini butuh **5-15 menit** tergantung internet.

**Yang diinstall:**
| Package | Fungsi |
|---|---|
| `tensorflow` | Framework untuk train LSTM |
| `google-generativeai` | LLM Gemini (claim extraction + evidence judge) |
| `chromadb` | Vector database untuk evidence retrieval |
| `sentence-transformers` | Embedding model untuk semantic search |
| `rank_bm25` | Keyword-based search |
| `fastapi` + `uvicorn` | Backend API server |
| `streamlit` | Web UI demo |
| `pydantic` | API request/response validation |
| `wikipedia-api` | Wikipedia fallback |
| `pandas`, `numpy`, `scikit-learn` | Data processing & evaluation |
| `nltk`, `sastrawi` | NLP Bahasa Indonesia |
| `beautifulsoup4`, `lxml` | Web scraping |

### 1.6 Buat Folder Struktur

```bash
mkdir data
mkdir data\raw
mkdir data\processed
mkdir data\training
mkdir data\evaluation
mkdir data\evidence
mkdir models
mkdir models\lstm
```

Kalau sudah ada, tidak apa-apa.

---

## 2. Setup API Key Gemini

### 2.1 Daftar / Login Google AI Studio

1. Buka: **https://aistudio.google.com/apikey**
2. Login dengan akun Google (Gmail)
3. Klik **"Create API Key"**
4. Pilih project (buat baru kalau diminta)
5. **Copy** API key yang muncul

### 2.2 Simpan ke File .env

Buka file `.env` di folder FAKTA:

```
FAKTA/
├── .env          ← Edit file ini
├── .env.example  ← Template (jangan edit)
└── ...
```

Isi dengan API key Anda:

```ini
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

> **Ganti `AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXX` dengan API key Anda yang sebenarnya.**

### 2.3 Verifikasi

Di terminal, cek apakah env terbaca:

**Windows CMD:**
```bash
type .env
```

**PowerShell:**
```bash
Get-Content .env
```

Pastikan baris `GEMINI_API_KEY=` terisi dengan key Anda.

### 2.4 (Opsional) Google Fact Check API

Ini opsional. Sistem tetap jalan tanpa ini.

1. Buka: https://console.cloud.google.com/
2. Buat project baru
3. Search "Fact Check Tools API" → Enable
4. Go to Credentials → Create API Key
5. Tambahkan ke `.env`:
   ```
   GOOGLE_FACTCHECK_API_KEY=AIzaSyYYYYYYYYYYYYYYYY
   ```

---

## 3. Kumpulkan Dataset

LSTM butuh dataset berlabel untuk belajar. Target: **minimum 5,000 sample**.

### 3.1 Opsi A: Scraping Otomatis (Recommended)

```bash
python src/data/collect.py
```

Script ini akan:
- Scrape TurnBackHoax.id (~50 halaman, ~1500+ artikel hoax)
- Simpan ke `data/raw/turnbackhoax/`
- Normalize ke format training

> ⚠️ Proses ini butuh **15-30 menit** karena rate limiting (1 request per 2 detik).

### 3.2 Opsi B: Download Dataset dari Kaggle

1. Buka https://www.kaggle.com
2. Search: **"indonesian fake news"** atau **"indonesian hoax"**
3. Download CSV
4. Simpan ke folder `data/training/`

Dataset yang direkomendasikan:

| Nama Dataset | URL | Jumlah | Label |
|---|---|---|---|
| Indonesian Fake News Detection | Kaggle | ~4,000-8,000 | Hoax/Non-hoax |
| ISHOX | Kaggle/Mendeley | ~3,000-5,000 | Hoax/Non-hoax |

### 3.3 Opsi C: Buat Dataset Manual

Kumpulkan artikel dari berbagai sumber dan buat file CSV:

```bash
# Buat file: data/training/manual.csv
```

Format file CSV:
```csv
text,label
"Viral! Matcha menyebabkan gagal ginjal!! Sebarkan!",hoax
"BMKG mencatat gempa magnitudo 5.2 di Maluku hari ini.",valid
"Konon harga BBM akan naik bulan depan, belum ada konfirmasi.",uncertain
"BPOM: Vaksin Covid-19 aman dan telah melalui uji klinis.",valid
"SEBARKAN!!! Air garam menyembuhkan semua penyakit!!!",hoax
```

**Sumber untuk label VALID (bukan hoax):**
- Kompas.com → section news → copy artikel
- Antara News → section berita
- BMKG.go.id → pengumuman resmi
- Kemenkes.go.id → statement resmi
- BPOM.go.id → pernyataan produk

**Sumber untuk label HOAX:**
- TurnBackHoax.id → judul + isi debunk
- MAFINDO → database hoaks

**Sumber untuk label UNCERTAIN:**
- Rumor yang belum dikonfirmasi
- Klaim yang masih berkembang
- Berita yang belum diverifikasi

### 3.4 Target Dataset

| Label | Minimum | Ideal | Sumber Utama |
|---|---|---|---|
| **hoax** | 3,000 | 10,000+ | TurnBackHoax, MAFINDO |
| **valid** | 2,000 | 5,000+ | Kompas, Tempo, BMKG, BPOM |
| **uncertain** | 200 | 1,000+ | Manual labeling |
| **TOTAL** | **5,200** | **16,000+** | |

> **PENTING:** Semakin banyak data → semakin akurat LSTM. Target **10,000+** untuk hasil bagus.

### 3.5 Verifikasi Dataset

Pastikan file CSV ada di `data/training/`:

```bash
dir data\training\
```

Harus ada minimal 1 file `.csv` dengan kolom `text` dan `label`.

---

## 4. Train Model LSTM

> **💡 REKOMENDASI: Pakai Google Colab (GRATIS + GPU)**
>
> Training di laptop bisa 15-30 menit. Di Colab dengan GPU cuma **2-5 menit**.
>
> File notebook Colab sudah ada di: `notebooks/colab_lstm_training.ipynb`

### Opsi A: Training di Google Colab (Recommended ⭐)

#### Langkah 1 — Buka Colab
1. Buka: **https://colab.research.google.com/**
2. Klik **Upload**
3. Pilih file: `notebooks/colab_lstm_training.ipynb`

#### Langkah 2 — Pilih GPU
1. Menu: **Runtime** → **Change runtime type**
2. Hardware accelerator: **T4 GPU**
3. Klik **Save**

#### Langkah 3 — Run Semua Cell
1. Klik cell pertama → **Shift+Enter** untuk run
2. Atau: **Runtime** → **Run all**
3. Ikuti instruksi di setiap cell

#### Langkah 4 — Upload Dataset di Colab
Saat cell upload muncul:
1. Klik **Choose Files**
2. Pilih file CSV dataset Anda
3. Tunggu upload selesai

#### Langkah 5 — Download Model Setelah Training
Di cell terakhir notebook Colab:
1. Run cell "Download Model"
2. File `lstm_model.zip` akan otomatis ter-download
3. Extract ke folder lokal:
   ```
   FAKTA/
   └── models/
       └── lstm/
           ├── lstm_model.keras    ← dari zip
           ├── tokenizer.pkl       ← dari zip
           └── label_map.json      ← dari zip
   ```

**Keuntungan Colab:**
- ✅ GPU gratis (training 5x lebih cepat)
- ✅ Tidak perlu install TensorFlow di laptop
- ✅ RAM 12GB+ (laptop biasanya 8GB)
- ✅ Semua dependency sudah tersedia
- ✅ Bisa share notebook ke orang lain

### Opsi B: Training di Laptop (Lokal)

```bash
python src/classifier/train_lstm.py data/training models/lstm
```

Atau via Jupyter Notebook (lebih interaktif):

```bash
jupyter notebook notebooks/02_lstm_training.ipynb
```

Lalu run semua cell satu per satu.

### 4.2 Yang Terjadi Saat Training

```
Step 1: Load semua CSV dari data/training/
Step 2: Cleaning + normalisasi text (lowercase, remove URL, slang)
Step 3: Tokenize → convert text ke angka sequences
Step 4: Pad sequences → semua text jadi panjang 200
Step 5: Split data → 70% train, 15% val, 15% test
Step 6: Hitung class weights (handle data imbalance)
Step 7: Build BiLSTM model
Step 8: Train (max 20 epochs, early stop kalau tidak improve 5 epoch)
Step 9: Evaluate pada test set
Step 10: Save model → models/lstm/lstm_model.keras
Step 11: Save tokenizer → models/lstm/tokenizer.pkl
```

Training time: **10-30 menit** tergantung jumlah data dan hardware.

### 4.3 Output Training

Di terminal akan muncul:

```
Loaded 8500 samples:
label
hoax       5200
valid      2800
uncertain   500

Train: 5950, Val: 1275, Test: 1275
Class weights: {0: 1.2, 1: 0.8, 2: 3.4}

Epoch 1/20
186/186 - loss: 0.8500 - accuracy: 0.6200 - val_loss: 0.7200 - val_accuracy: 0.6800
...
Epoch 12/20
186/186 - loss: 0.2100 - accuracy: 0.9100 - val_loss: 0.3500 - val_accuracy: 0.8500

Test Loss: 0.3800, Test Accuracy: 0.8400

Classification Report:
              precision    recall  f1-score
      valid       0.82      0.85      0.83
       hoax       0.88      0.82      0.85
  uncertain       0.65      0.60      0.62

Model saved to models/lstm/lstm_model.keras
Tokenizer saved to models/lstm/tokenizer.pkl
```

### 4.4 Target Metrik Training

| Metrik | Minimum | Bagus | Sangat Bagus |
|---|---|---|---|
| Accuracy | > 0.75 | > 0.85 | > 0.90 |
| F1-Score Hoax | > 0.70 | > 0.80 | > 0.88 |
| F1-Score Valid | > 0.70 | > 0.80 | > 0.88 |

### 4.5 Jika Accuracy Rendah

| Masalah | Solusi |
|---|---|
| **Dataset terlalu kecil** | Tambah data: scrape lebih banyak, download dari Kaggle |
| **Imbalanced classes** | Tambah sample untuk kelas minoritas |
| **Overfitting** (train acc tinggi, val acc rendah) | Naikkan `dropout_rate` ke 0.5 di `configs/lstm_config.yaml` |
| **Underfitting** (train acc rendah) | Naikkan `lstm_units` ke 128, `epochs` ke 30 |
| **Tidak converge** | Turunkan `learning_rate` ke 0.0005 |

Edit config di `configs/lstm_config.yaml`:
```yaml
lstm_units: 128       # default: 64, naikkan kalau underfit
dropout_rate: 0.5     # default: 0.3, naikkan kalau overfit
epochs: 30            # default: 20
```

### 4.6 Test LSTM Setelah Training

```bash
python src/classifier/predict_lstm.py "VIRAL!!! Matcha menyebabkan gagal ginjal!! Sebarkan!!!"
```

Expected output:
```
FAKTA LSTM Prediction:
  Hoax:     0.8234
  Valid:    0.1200
  Uncertain: 0.0566

  Verdict: HOAX
```

Test dengan teks valid:
```bash
python src/classifier/predict_lstm.py "BMKG mencatat gempa magnitudo 5.2 di Maluku pada 15 Januari 2025."
```

Expected output:
```
FAKTA LSTM Prediction:
  Hoax:     0.1500
  Valid:    0.7800
  Uncertain: 0.0700

  Verdict: TIDAK HOAX
```

---

## 5. Setup Evidence Retrieval

Evidence retrieval mencari bukti dari sumber kredibel untuk memverifikasi klaim.

### 5.1 Index Evidence ke Database Lokal

Artikel debunk dari TurnBackHoax digunakan sebagai evidence base:

```bash
python src/evidence/indexer.py
```

Proses ini akan:
- Membaca debunk articles dari `data/processed/`
- Generate embeddings (semantic representation)
- Index ke ChromaDB (local vector database)
- Build BM25 index (keyword search)

> ⚠️ Pertama kali akan download model embedding (~300MB). Butuh internet.

### 5.2 Test Retrieval

```bash
python src/evidence/retriever.py
```

Expected output:
```
[Retriever] ChromaDB initialized at data/evidence/chroma_db
[Retriever] Collection has 2 documents
[Retriever] Sentence encoder initialized

Search results for 'matcha gagal ginjal':
  [BPOM] score=0.823 - BPOM menyatakan bahwa matcha aman dikonsumsi...

Database stats: {'chroma_count': 2, 'bm25_count': 2}
```

### 5.3 Test Evidence Judge (Butuh API Key)

```bash
python src/judge/gemini_evidence_judge.py
```

Expected output (jika API key benar):
```
Verdict: Refuted
Confidence: 0.88
Reasoning: Evidence dari Kemenkes membantah klaim bahwa matcha menyebabkan gagal ginjal.
```

---

## 6. Jalankan Sistem

Sistem FAKTA punya 2 komponen yang dijalankan:
1. **API Server** (FastAPI) — backend
2. **Demo UI** (Streamlit) — frontend

### 6.1 Start API Server

Dari folder root FAKTA:

```bash
cd src/api
python main.py
```

Atau dari root:
```bash
python -m src.api.main
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Initializing FAKTA pipeline...
INFO:     LSTM model loaded
INFO:     FAKTA pipeline initialized
```

> ⚠️ Biarkan terminal ini tetap terbuka. Jangan ditutup.

### 6.2 Test API

Buka **terminal BARU** (jangan tutup terminal API):

```bash
# Health check
curl http://localhost:8000/
```

Expected response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "lstm": "loaded",
    "retriever": "initialized",
    "judge": "initialized"
  }
}
```

Test check article:
```bash
curl -X POST http://localhost:8000/check ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"VIRAL!!! Matcha menyebabkan gagal ginjal!! Sebarkan sebelum dihapus!!\"}"
```

*(Untuk PowerShell, ganti `^` dengan `` ` ``)*

Expected response:
```json
{
  "verdict": "Hoax",
  "confidence": 0.72,
  "avg_hoax_score": 0.78,
  "summary": "Artikel mengandung klaim yang dibantah...",
  "claims": [...],
  "processing_time_ms": 3420
}
```

### 6.3 Start Demo UI (Terminal Baru)

```bash
streamlit run app/streamlit_app.py
```

Browser akan otomatis buka ke: **http://localhost:8501**

### 6.4 Cara Pakai UI

1. **Paste teks berita** di textarea
2. **Klik "Periksa"**
3. Lihat hasil:
   - ✅ **Tidak Hoax** — klaim didukung evidence
   - ❌ **Hoax** — klaim dibantah evidence
   - ⚠️ **Tidak Cukup Bukti** — evidence tidak cukup
4. Klik **expand** pada setiap klaim untuk lihat detail:
   - LSTM hoax probability
   - LLM verdict + confidence
   - Evidence sources yang digunakan
   - Reasoning dari LLM

### 6.5 Contoh Teks untuk Dicoba

**Hoax:**
```
VIRAL!!! Matcha menyebabkan gagal ginjal dan sudah banyak korban meninggal!!
Sebarkan sebelum dihapus!! Obat ini disembunyikan oleh pemerintah!!
```

**Valid:**
```
BMKG mencatat gempa magnitudo 5.2 di Maluku pada tanggal 15 Januari 2025.
Gempa tidak berpotensi tsunami. Warga dihimbau tetap tenang.
```

**Uncertain:**
```
Kabar beredar bahwa harga BBM akan naik bulan depan.
Pemerintah belum memberikan konfirmasi resmi mengenai hal ini.
```

---

## 7. Testing & Evaluasi

### 7.1 Test Fusion Engine (Tidak Butuh API Key)

```bash
python src/fusion/confidence_fusion.py
```

Expected output — 4 skenario:
```
======================================================================
FAKTA Confidence Fusion Engine — DEMO
======================================================================

--- Example 1: Strong evidence, claim refuted ---
  Final hoax score: 0.7065
  Confidence:       0.89
  Verdict:          Hoax
  Mode:             strong_evidence

--- Example 2: LSTM suspects hoax, no evidence ---
  Final hoax score: 0.5472
  Confidence:       0.42
  Verdict:          Tidak Cukup Bukti
  Mode:             no_evidence

--- Example 3: Strong evidence SUPPORTS claim (valid) ---
  Final hoax score: 0.2785
  Confidence:       0.85
  Verdict:          Tidak Hoax
  Mode:             strong_evidence
  NOTE: Despite LSTM=0.85, evidence overrides → Tidak Hoax!

--- Example 4: Conflicting evidence ---
  Final hoax score: 0.5015
  Confidence:       0.476
  Verdict:          Tidak Cukup Bukti
  Mode:             strong_evidence
```

### 7.2 Test Preprocessing (Tidak Butuh API Key)

```bash
python src/preprocessing/cleaning.py
python src/preprocessing/feature_extractor.py
```

### 7.3 Test Claim Extraction (Fallback Mode, Tanpa API)

```bash
python src/claim_extraction/gemini_extractor.py
```

Expected output (menggunakan fallback rule-based):
```
  [causal] Viral di media sosial! Matcha menyebabkan gagal ginjal dan sudah banyak korban meninggal (importance: 1.0)
  [attribution] Menurut dr (importance: 0.5)
  [factual] Namun BPOM membantah klaim tersebut dan menyatakan matcha aman dikonsumsi (importance: 0.5)
```

### 7.4 End-to-End Test via Notebook

```bash
jupyter notebook notebooks/04_end_to_end_evaluation.ipynb
```

Run semua cell untuk test full pipeline.

### 7.5 Buat Dataset Evaluasi

Buat file `data/evaluation/test_claims.csv`:

```csv
text,label_ground_truth
"BPOM: Matcha aman dikonsumsi dalam jumlah wajar",valid
"Vaksin mengandung chip 5G, sebarkan!",hoax
"BMKG: Tidak ada gempa di Jakarta hari ini",valid
"Konon harga BBM akan naik, belum ada konfirmasi",uncertain
"Air garam menyembuhkan kanker, WHO sudah konfirmasi",hoax
"Pemerintah resmi menaikkan pajak PPN jadi 12%",valid
```

Test dengan API:
```bash
curl -X POST http://localhost:8000/check ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"Air garam menyembuhkan kanker, WHO sudah konfirmasi\"}"
```

Cek apakah verdict sesuai dengan `label_ground_truth`.

---

## 8. Troubleshooting

| Error | Penyebab | Solusi |
|---|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | Dependencies belum install | `pip install -r requirements.txt` |
| `Model not found at models/lstm/lstm_model.keras` | Belum training LSTM | Jalankan training: `python src/classifier/train_lstm.py data/training models/lstm` |
| `GEMINI_API_KEY not set` | API key belum di `.env` | Buat file `.env` dan isi `GEMINI_API_KEY=your_key` |
| `ChromaDB init failed` | Folder `data/evidence/` belum ada | `mkdir data\evidence` |
| `Rate limit exceeded` | Terlalu banyak API call ke Gemini | Tunggu 1 menit. Sistem punya rate limiter otomatis |
| `Port 8000 already in use` | Ada program lain di port 8000 | Edit `src/api/main.py`, ganti `port=8000` jadi `port=8001` |
| `Port 8501 already in use` | Streamlit sudah jalan di terminal lain | Buka http://localhost:8501 di browser, atau kill process |
| `CUDA error` / GPU issue | TensorFlow butuh GPU tapi tidak ada | Install `tensorflow-cpu` bukan `tensorflow`: `pip uninstall tensorflow && pip install tensorflow-cpu` |
| `ConnectionError` saat scraping | Internet tidak stabil / site down | Coba lagi nanti, atau download dataset manual dari Kaggle |
| `UnicodeDecodeError` saat load CSV | File CSV encoding bukan UTF-8 | Buka CSV di Excel, save as → UTF-8 |
| `PermissionError` di Windows | Terminal tidak punya hak akses | Run terminal sebagai Administrator |

### 8.1 Debug Mode

Kalau ada error yang tidak jelas, tambahkan print statement di `src/api/main.py`:

```python
# Di dalam method process(), tambahkan:
print(f"DEBUG: lstm_proba = {lstm_proba}")
print(f"DEBUG: evidence count = {len(evidence)}")
print(f"DEBUG: llm_verdict = {llm_verdict}")
print(f"DEBUG: fusion result = {fusion_result}")
```

### 8.2 Cek Status Komponen

```bash
curl http://localhost:8000/stats
```

Response:
```json
{
  "pipeline_initialized": true,
  "cache_stats": {"total_entries": 0},
  "retriever_stats": {"chroma_count": 2, "bm25_count": 2}
}
```

---

## 📌 Ringkasan Command

```bash
# === SETUP (sekali saja) ===
python -m venv venv                    # Buat virtual env
venv\Scripts\activate                  # Activate (Windows)
pip install -r requirements.txt        # Install dependencies

# === DATA ===
python src/data/collect.py             # Scrape TurnBackHoax (15-30 menit)

# === TRAINING (sekali, atau ulang kalau data berubah) ===
python src/classifier/train_lstm.py data/training models/lstm

# === EVIDENCE (sekali, atau ulang kalau data berubah) ===
python src/evidence/indexer.py

# === JALANKAN SETIAP HARI ===
python src/api/main.py                 # Terminal 1: API server
streamlit run app/streamlit_app.py     # Terminal 2: Demo UI

# === TESTING ===
python src/fusion/confidence_fusion.py         # Test fusion engine
python src/preprocessing/cleaning.py            # Test preprocessing
python src/classifier/predict_lstm.py "text"    # Test LSTM prediction
python src/evidence/retriever.py                # Test evidence retrieval
```

---

## ⏱️ Estimasi Waktu

| Tahap | Waktu | Keterangan |
|---|---|---|
| Setup environment | 10-15 menit | Install dependencies (lokal) |
| Setup API key | 5 menit | Daftar + copy key |
| Data collection | 30 menit - 2 jam | Scraping atau download |
| **Training LSTM (Colab + GPU)** | **2-5 menit** | ⭐ Recommended |
| **Training LSTM (Lokal/CPU)** | **10-30 menit** | Lebih lambat |
| Setup evidence | 5-10 menit | Indexing |
| Start API + UI | 2 menit | Langsung jalan |
| **TOTAL (dengan Colab)** | **~45 menit - 2 jam** | Setup awal |

---

## 🆚 Colab vs Lokal — Mana yang Harus Dipilih?

| Fitur | Google Colab | Laptop Lokal |
|---|---|---|
| **Kecepatan training** | 2-5 menit (GPU T4) | 10-30 menit (CPU) |
| **Install TensorFlow** | ❌ Tidak perlu | ✅ Perlu |
| **RAM** | 12GB+ | Tergantung laptop (biasanya 8GB) |
| **Disk space** | Tidak makan storage laptop | Butuh ~1-2GB untuk TensorFlow |
| **Internet** | Perlu (untuk upload/run) | Tidak perlu saat training |
| **Biaya** | GRATIS | GRATIS |
| **Recommended?** | ⭐ YA | Kalau laptop powerful |

**Kesimpulan:** Pakai **Google Colab** untuk training. Setelah selesai, download model dan jalankan sistem di laptop lokal.

---

## 💡 Tips untuk Demo/Presentasi

1. **Pre-load 3 contoh** yang pasti berhasil:
   - 1 hoax (hasil: ❌ Hoax)
   - 1 valid (hasil: ✅ Tidak Hoax)
   - 1 uncertain (hasil: ⚠️ Tidak Cukup Bukti)

2. **Screenshot hasil** sebelum presentasi — jaga-jaga kalau API down

3. **Jelaskan arsitektur** dengan diagram — highlight fusion engine sebagai kontribusi utama

4. **Tunjukkan perbandingan:**
   - LSTM-only → hanya tebak dari gaya bahasa
   - Full hybrid → LSTM + evidence → lebih akurat

5. **Siapkan fallback:** kalau API down, fusion engine bisa di-test dengan mock data:
   ```bash
   python src/fusion/confidence_fusion.py
   ```
