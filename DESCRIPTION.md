# 🏛️ FAKTA — Deskripsi Lengkap Sistem

**FAKTA** (Fact-Checking AI) adalah sistem pendeteksi hoaks bahasa Indonesia berbasis **Hybrid LSTM + LLM + Evidence**. Sistem ini menggabungkan 3 pendekatan berbeda untuk menghasilkan verdict yang lebih robust daripada menggunakan salah satu pendekatan saja.

---

## 🎯 Masalah yang Dipecahkan

Deteksi hoaks di bahasa Indonesia itu sulit karena:
- **Hoax ditulis mirip berita asli** — sulit dibedakan hanya dari konten
- **LLM bisa hallucinate** — tidak bisa dipercaya 100% tanpa verifikasi
- **Cek manual butuh waktu lama** — tidak skalabel

FAKTA menjawab ini dengan **hybrid approach**: LSTM (statistik/gaya bahasa) + LLM (reasoning) + Evidence (fakta nyata).

---

## 🏗️ Arsitektur Lengkap

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. USER INPUT                                                       │
│    "VIRAL!!! Matcha menyebabkan gagal ginjal!! Sebarkan!!"          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING (src/preprocessing/)                               │
│    - clean_text()           → hapus URL, mention, special char      │
│    - normalize_slang()      → "gak" → "tidak", "yg" → "yang"       │
│    - extract_features()     → hitung caps_ratio, exclamation_count, │
│                               provocative_word_count, dll (14 fitur)│
│    Output: "viral matcha menyebabkan gagal ginjal sebarkan"         │
│            + linguistic_hoax_score (0.0 - 1.0)                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. CLAIM EXTRACTION (src/claim_extraction/gemini_extractor.py)      │
│                                                                     │
│    Input: teks yang sudah dibersihkan                               │
│    API Call: Gemini 2.0 Flash                                       │
│    Prompt: "Ekstrak klaim faktual dari teks berikut"                │
│                                                                     │
│    Output: List<Claim>                                              │
│    - Claim 1: "Matcha menyebabkan gagal ginjal" (causal)            │
│    - Claim 2: "Sudah banyak korban meninggal" (factual)             │
│    - Claim 3: "Obat ini disembunyikan pemerintah" (attribution)     │
│    (claim tipe "opinion" di-skip karena tidak bisa dicek)           │
│                                                                     │
│    Fallback: Kalau API down → regex-based extraction                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    Untuk SETIAP CLAIM:
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌─────────────────────────┐     ┌───────────────────────────────────────┐
│ 4a. LSTM CLASSIFIER     │     │ 4b. EVIDENCE RETRIEVAL                │
│ (src/classifier/)       │     │ (src/evidence/)                       │
│                         │     │                                       │
│ Input: cleaned text      │     │ Input: claim text                     │
│ Model: BiLSTM trained    │     │                                       │
│ pada 16K sample hoax/   │     │ ① Google Fact Check API               │
│ valid                   │     │    (src/evidence/factcheck_api.py)     │
│                         │     │    → cari fact-check dari web          │
│ Output: {hoax: 0.85,    │     │                                       │
│          valid: 0.10,   │     │ ② Local ChromaDB + BM25               │
│          uncertain:0.05}│     │    (src/evidence/retriever.py)         │
│                         │     │    → hybrid: semantic + keyword search │
│ linguistic_hoax_score   │     │    → dari debunk articles TurnBackHoax │
│ dari feature_extractor  │     │                                       │
│ juga dihitung           │     │ ③ Wikipedia Fallback                  │
│                         │     │    (src/evidence/wikipedia_fallback.py)│
│ ⚠️ LSTM ini "insting" — │     │    → kalau evidence < 2, cari di Wiki  │
│   cuma liat GAYA BAHASA │     │                                       │
│   bukan ISI FAKTA       │     │ Output: List<Evidence>                 │
│                         │     │   [BPOM: "Matcha aman..."] score=0.82 │
└────────────┬────────────┘     │   [Kemenkes: "..."] score=0.75         │
             │                  └────────────┬──────────────────────────┘
             │                               │
             │                               ▼
             │                  ┌─────────────────────────────────────┐
             │                  │ 4c. LLM EVIDENCE JUDGE              │
             │                  │ (src/judge/gemini_evidence_judge.py)│
             │                  │                                     │
             │                  │ Input: claim + evidence list        │
             │                  │ API Call: Gemini 2.0 Flash          │
             │                  │                                     │
             │                  │ Prompt:                             │
             │                  │ "Bandingkan klaim ini dengan bukti. │
             │                  │  Apakah klaim didukung, dibantah,   │
             │                  │  atau tidak cukup bukti?"           │
             │                  │                                     │
             │                  │ Output:                             │
             │                  │   llm_verdict: "Supported"/"Refuted"│
             │                  │   llm_confidence: 0.88              │
             │                  │   reasoning: "BPOM menyatakan..."   │
             └──────┬───────────┘                                     │
                    │                                                 │
                    ▼                                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. CONFIDENCE FUSION (src/fusion/confidence_fusion.py)                  │
│                                                                         │
│  Menggabungkan semua sinyal jadi satu verdict per claim:                │
│                                                                         │
│  Input:                                                                 │
│    - lstm_hoax_proba    = 0.85  (dari LSTM)                             │
│    - llm_verdict        = "Refuted" (dari Gemini Judge)                 │
│    - llm_confidence     = 0.88  (dari Gemini Judge)                     │
│    - evidence_quality   = 0.82  (dari retriever, berdasarkan score)     │
│    - linguistic_hoax    = 0.75  (dari feature extractor)                │
│                                                                         │
│  3 Fusion Regime:                                                       │
│  ┌─────────────────────┬───────────────────────────────────────────┐   │
│  │ Strong Evidence     │ LLM + evidence dominan. LSTM diabaikan.   │   │
│  │ (evidence_quality   │ Final = llm_confidence × direction         │   │
│  │  > 0.7)             │                                            │   │
│  ├─────────────────────┼───────────────────────────────────────────┤   │
│  │ Weak Evidence       │ Weighted average: LLM 50%, LSTM 30%,      │   │
│  │ (0.3 < eq <= 0.7)   │ linguistic 20%                            │   │
│  ├─────────────────────┼───────────────────────────────────────────┤   │
│  │ No Evidence         │ LSTM 60%, linguistic 40%.                 │   │
│  │ (eq <= 0.3)         │ Confidence diturunkan (max 0.5)           │   │
│  └─────────────────────┴───────────────────────────────────────────┘   │
│                                                                         │
│  Output per claim:                                                      │
│    - verdict: "Hoax" / "Tidak Hoax" / "Tidak Cukup Bukti"               │
│    - final_hoax_score: 0.71                                             │
│    - confidence: 0.89                                                   │
│    - mode: "strong_evidence"                                            │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. ARTICLE AGGREGATION (src/fusion/aggregation.py)                  │
│                                                                     │
│  Menggabungkan verdict semua claim jadi 1 verdict artikel:          │
│  - Weighted average berdasarkan importance tiap claim               │
│  - Kalau ada 1 claim "Hoax" dengan confidence tinggi → artikel Hoax │
│  - Kalau semua claim "Tidak Cukup Bukti" → artikel NEI              │
│                                                                     │
│  Output:                                                            │
│    - verdict: "Hoax"                                                │
│    - confidence: 0.72                                               │
│    - avg_hoax_score: 0.78                                           │
│    - summary: "Artikel mengandung klaim yang dibantah..."           │
│    - claims: [detail per claim]                                     │
│    - processing_time_ms: 3420                                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 7. RESPONSE ke USER                                                 │
│                                                                     │
│  Via FastAPI (port 8000) atau Streamlit UI (port 8501)              │
│                                                                     │
│  {                                                                  │
│    "verdict": "Hoax",                                               │
│    "confidence": 0.72,                                              │
│    "claims": [                                                      │
│      {                                                              │
│        "claim_text": "Matcha menyebabkan gagal ginjal",             │
│        "verdict": "Hoax",                                           │
│        "confidence": 0.89,                                          │
│        "lstm_hoax_proba": 0.85,                                     │
│        "llm_verdict": "Refuted",                                    │
│        "llm_confidence": 0.88,                                      │
│        "evidence_sources": ["BPOM", "Kemenkes"],                    │
│        "reasoning": "Evidence dari BPOM membantah klaim..."         │
│      }                                                              │
│    ]                                                                │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Penjelasan Tiap Komponen

### 🧹 Preprocessing (`src/preprocessing/`)

| File | Fungsi |
|---|---|
| `cleaning.py` | Membersihkan teks: lowercase, hapus URL, mention (@), hashtag (#), special character, normalisasi whitespace |
| `slang_normalizer.py` | Mengganti slang bahasa Indonesia ke formal: "gak" → "tidak", "yg" → "yang", "gw" → "saya", dll (50+ mapping) |
| `feature_extractor.py` | Mengekstrak 14 fitur linguistik dari teks: |

**14 Fitur Linguistik:**
1. `caps_ratio` — proporsi huruf kapital
2. `exclamation_count` — jumlah tanda seru (!)
3. `question_count` — jumlah tanda tanya (?)
4. `provocative_word_count` — kata provokatif ("viral", "geger", "heboh")
5. `urgency_word_count` — kata urgensi ("segera", "sebarkan")
6. `avg_word_length` — rata-rata panjang kata
7. `unique_word_ratio` — rasio kata unik
8. `sentence_count` — jumlah kalimat
9. `avg_sentence_length` — rata-rata panjang kalimat
10. `quote_count` — jumlah kutipan
11. `number_count` — jumlah angka
12. `ellipsis_count` — jumlah titik tiga (...)
13. `repetition_count` — kata yang diulang
14. `emotional_word_count` — kata emosional

### 📢 Claim Extraction (`src/claim_extraction/`)

**File:** `gemini_extractor.py`

**Fungsi:** Mengekstrak klaim faktual dari teks menggunakan Gemini 2.0 Flash.

**Jenis klaim yang diekstrak:**
- `causal` — klaim sebab-akibat ("X menyebabkan Y")
- `factual` — klaim fakta ("Terjadi Z pada tanggal T")
- `attribution` — klaim atribusi ("Menurut A, B terjadi")
- `opinion` — pendapat (di-skip, tidak bisa dicek fakta)

**Fallback:** Kalau API Gemini down, pakai regex-based extraction yang mendeteksi pola klaim sederhana.

### 🤖 LSTM Classifier (`src/classifier/`)

| File | Fungsi |
|---|---|
| `lstm_model.py` | Definisi model BiLSTM + class `LSTMPredictor` untuk inference |
| `predict_lstm.py` | CLI untuk prediksi langsung: `python predict_lstm.py "teks"` |
| `train_lstm.py` | Script training: `python train_lstm.py data/training models/lstm` |

**Arsitektur Model:**
```
Embedding (vocab=20000, dim=128)
  → Dropout (0.3)
    → BiLSTM (64 units, return_sequences=True)
      → BiLSTM (32 units, return_sequences=False)
        → Dense (32, relu)
          → Dropout (0.3)
            → Dense (3, softmax) → [valid, hoax, uncertain]
```

**Dataset Training:**
- 16,474 sample (12,777 hoax + 3,697 valid)
- Sumber: Kaggle Indonesian Fake News Dataset + TurnBackHoax API (1,000 artikel)
- Split: 70% train, 15% validation, 15% test

**⚠️ Penting:** LSTM ini hanya melihat **gaya bahasa**, bukan isi fakta. Dia tidak tahu apakah Matcha benar-benar menyebabkan gagal ginjal — dia hanya mengenali pola tulisan yang mirip hoax (banyak tanda seru, kata emosional, huruf kapital, dll).

### 🔍 Evidence Retrieval (`src/evidence/`)

**3 sumber bukti, dipanggil berurutan:**

| Sumber | File | Fungsi |
|---|---|---|
| ① Google Fact Check API | `factcheck_api.py` | Mencari fact-check dari web menggunakan Google Fact Check Tools API |
| ② Local ChromaDB + BM25 | `retriever.py` | Hybrid search: semantic (ChromaDB vector) + keyword (BM25) dari database debunk TurnBackHoax |
| ③ Wikipedia Fallback | `wikipedia_fallback.py` | Kalau evidence dari sumber 1 & 2 kurang dari 2, cari di Wikipedia |

**ChromaDB:** Vector database lokal yang menyimpan embedding dari debunk articles. Dicocokkan menggunakan semantic similarity.

**BM25:** Keyword-based search yang melacak kata-kata spesifik dalam claim.

Kedua hasil digabung dan di-ranking berdasarkan relevance score.

**Caching (`cache.py`):** Evidence disimpan di cache supaya pencarian berulang untuk claim yang sama tidak perlu diulang — mempercepat response time.

### ⚖️ LLM Evidence Judge (`src/judge/`)

**File:** `gemini_evidence_judge.py`

**Fungsi:** Membandingkan klaim dengan bukti yang ditemukan.

**Input:**
- Claim text: "Matcha menyebabkan gagal ginjal"
- Evidence list: [BPOM: "Matcha aman dikonsumsi", Kemenkes: "..."]

**API Call:** Gemini 2.0 Flash

**Prompt (internal):** "Bandingkan klaim ini dengan bukti-bukti berikut. Apakah klaim didukung, dibantah, atau tidak cukup bukti?"

**Output:**
```json
{
  "llm_verdict": "Refuted",
  "llm_confidence": 0.88,
  "reasoning": "Evidence dari BPOM dan Kemenkes membantah klaim bahwa matcha menyebabkan gagal ginjal."
}
```

### 🔀 Confidence Fusion (`src/fusion/`)

**File:** `confidence_fusion.py`

**Ini adalah inti/core dari FAKTA** — komponen yang menggabungkan semua sinyal menjadi satu verdict.

**Input:**
| Sinyal | Sumber | Contoh Nilai |
|---|---|---|
| `lstm_hoax_proba` | LSTM Classifier | 0.85 |
| `llm_verdict` | Gemini Evidence Judge | "Refuted" |
| `llm_confidence` | Gemini Evidence Judge | 0.88 |
| `evidence_quality` | Evidence Retriever | 0.82 |
| `linguistic_hoax` | Feature Extractor | 0.75 |

**3 Fusion Regime:**

| Regime | Kondisi | Cara Hitung | Contoh |
|---|---|---|---|
| **Strong Evidence** | `evidence_quality > 0.7` | LLM + evidence dominan. LSTM diabaikan. | Evidence dari BPOM sangat jelas → verdict "Hoax" |
| **Weak Evidence** | `0.3 < evidence_quality ≤ 0.7` | Weighted average: LLM 50%, LSTM 30%, linguistic 20% | Evidence ada tapi tidak spesifik → blend semua sinyal |
| **No Evidence** | `evidence_quality ≤ 0.3` | LSTM 60%, linguistic 40%. Confidence max 0.5 | Tidak ada evidence → fallback ke insting |

**Output per claim:**
```json
{
  "verdict": "Hoax",
  "final_hoax_score": 0.71,
  "confidence": 0.89,
  "mode": "strong_evidence"
}
```

### 📦 Article Aggregation (`src/fusion/aggregation.py`)

**Fungsi:** Menggabungkan verdict semua claim dalam satu artikel menjadi verdict final.

**Logika:**
- Weighted average berdasarkan importance tiap claim
- Kalau ada 1 claim "Hoax" dengan confidence tinggi → artikel jadi "Hoax"
- Kalau semua claim "Tidak Cukup Bukti" → artikel "Tidak Dapat Diverifikasi"

**Output:**
```json
{
  "verdict": "Hoax",
  "confidence": 0.72,
  "avg_hoax_score": 0.78,
  "summary": "Artikel mengandung klaim yang dibantah oleh sumber kredibel.",
  "claim_stats": {
    "total_claims": 3,
    "verifiable_claims": 2,
    "hoax_claims": 1,
    "valid_claims": 0,
    "uncertain_claims": 1
  }
}
```

---

## 🔄 Fallback System

| Komponen Missing | Yang Terjadi |
|---|---|
| LSTM model tidak ditemukan | Pakai default: hoax=0.5, valid=0.25, uncertain=0.25 |
| Gemini API down | Claim extraction pakai regex fallback |
| Evidence retrieval kosong | Judge return "NotEnoughEvidence" |
| Semua komponen down | Verdict: "Tidak dapat diverifikasi" |

Sistem dirancang agar **selalu menghasilkan response** meskipun beberapa komponen gagal.

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|---|---|
| **ML (LSTM)** | TensorFlow/Keras 2.21, scikit-learn 1.9 |
| **LLM** | Google Gemini 2.0 Flash |
| **Retrieval** | ChromaDB + BM25 + sentence-transformers |
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit |
| **Data Processing** | Pandas, NumPy, NLTK, Sastrawi |
| **Web Scraping** | BeautifulSoup4, lxml, requests |

---

## 🌐 API Endpoints

| Method | Path | Deskripsi |
|---|---|---|
| `GET` | `/` | Health check — status komponen sistem |
| `POST` | `/check` | Periksa artikel — input teks, output verdict + detail per claim |
| `POST` | `/feedback` | Kirim feedback manusia pada verdict sistem |
| `GET` | `/stats` | Statistik sistem: cache, retriever, pipeline status |

---

## 📊 3 Pendekatan yang Digabung

| Pendekatan | Peran | Kelebihan | Kelemahan |
|---|---|---|---|
| **LSTM** | Deteksi gaya bahasa hoax | Cepat, tidak butuh internet, gratis | Tidak cek fakta, cuma style |
| **LLM (Gemini)** | Reasoning & claim extraction | Paham konteks, bisa reason | Bisa hallucinate, butuh API key |
| **Evidence** | Bukti nyata dari sumber kredibel | Berbasis fakta, bisa ditelusuri | Terbatas pada data yang ada |

**Yang membuat FAKTA berbeda:** fusion engine yang tahu kapan harus percaya komponen mana. Kalau evidence kuat → percaya evidence. Kalau evidence lemah → blend semua sinyal. Ini lebih robust daripada menggunakan salah satu pendekatan saja.

---

## 📁 Struktur Project

```
FAKTA/
├── src/
│   ├── api/
│   │   ├── main.py                  # FastAPI server + pipeline orchestration
│   │   └── schemas.py               # Pydantic request/response models
│   ├── classifier/
│   │   ├── lstm_model.py            # BiLSTM model definition + predictor
│   │   ├── predict_lstm.py          # CLI inference
│   │   └── train_lstm.py            # Training script
│   ├── claim_extraction/
│   │   └── gemini_extractor.py      # LLM claim extraction
│   ├── evidence/
│   │   ├── retriever.py             # Hybrid BM25 + ChromaDB search
│   │   ├── factcheck_api.py         # Google Fact Check API
│   │   ├── wikipedia_fallback.py    # Wikipedia search fallback
│   │   ├── cache.py                 # Evidence caching
│   │   └── indexer.py               # Evidence database indexer
│   ├── judge/
│   │   └── gemini_evidence_judge.py # LLM evidence comparison
│   ├── fusion/
│   │   ├── confidence_fusion.py     # Multi-signal fusion engine
│   │   └── aggregation.py           # Article-level aggregation
│   ├── preprocessing/
│   │   ├── cleaning.py              # Text cleaning
│   │   ├── slang_normalizer.py      # Indonesian slang normalization
│   │   └── feature_extractor.py     # 14 linguistic features
│   └── data/
│       └── collect.py               # Data collection (TurnBackHoax API)
├── app/
│   └── streamlit_app.py             # Web UI demo
├── data/
│   ├── raw/turnbackhoax/            # Raw scraped articles
│   ├── training/                    # Training datasets (CSV)
│   ├── evidence/                    # ChromaDB + BM25 index
│   └── evaluation/                  # Test data + feedback
├── models/
│   └── lstm/                        # Trained LSTM model + tokenizer
├── notebooks/
│   ├── colab_lstm_training.ipynb    # Google Colab training notebook
│   └── ...                          # Other analysis notebooks
├── configs/
│   └── lstm_config.yaml             # LSTM hyperparameters
├── .env                             # API keys (GEMINI_API_KEY)
├── requirements.txt                 # Python dependencies
├── README.md                        # Quick overview
├── PANDUAN_LENGKAP.md               # Step-by-step setup guide
└── DESKRIPSI_SISTEM.md              # ← File ini
```
