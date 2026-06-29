# Arsitektur Sistem FAKTA v3 — Fact-Checking AI Hybrid LSTM + LLM + Evidence

> **Tujuan:** Dokumen arsitektur lengkap untuk sistem pendeteksi hoaks Bahasa Indonesia
> **Versi:** 3.0 (Binary LSTM — Uncertain = lack of evidence, handled di fusion)

---

## 1. Gambaran Umum

FAKTA adalah sistem pendeteksi hoaks **hybrid** yang menggabungkan tiga pendekatan:

| Komponen | Peran |
|---|---|
| **LSTM Classifier** | Binary classification pola linguistik (hoax vs valid) |
| **LLM (Gemini)** | Ekstraksi klaim, reasoning berbasis evidence, verdict per klaim |
| **Evidence Retrieval** | Pencarian bukti dari Google Fact Check API, ChromaDB (BM25+embedding), Wikipedia |

Keputusan final diambil melalui **confidence fusion berbasis regime** — bukan linear weighted scoring biasa.
Evidence quality menjadi **confidence multiplier**, bukan skor hoax independen.

**Prinsip utama:** "Tidak Cukup Bukti" (uncertain) **bukan** sesuatu yang dipelajari LSTM.
Uncertain adalah **keadaan evidence** — terjadi ketika tidak ada evidence yang ditemukan untuk sebuah klaim.
Fusion engine secara otomatis menarik skor ke tengah (0.5) saat evidence_quality = 0.

### Perubahan dari Arsitektur v1

1. **LSTM menggantikan IndoBERT** — lebih ringan, cocok untuk deployment demo
2. **Binary LSTM (hoax vs valid)** — bukan 3-class. Uncertain = lack of evidence, bukan linguistic pattern
3. **Fusion formula direvisi total** — evidence quality = confidence multiplier, bukan skor hoax additive
4. **Double counting source credibility dihapus** — hanya dihitung sekali di evidence_quality
5. **LSTM pakai full article text** — bukan hanya claim text, agar pola gaya bahasa hoaks terbaca
6. **Article aggregation weighted** — bukan "1 hoax = artikel hoax", tapi importance + type + proportion
7. **NEI otomatis dari fusion** — evidence_quality = 0 → skor ditarik ke 0.5 → "Tidak Cukup Bukti"
8. **Evidence retrieval konkret** — Google FC API + ChromaDB + BM25 + Wikipedia tiered
9. **Training di Google Colab** — GPU gratis, lebih cepat 5-10x dari CPU lokal

---

## 2. Diagram Arsitektur

```
┌─────────────────────────────────────────────┐
│              USER INPUT                       │
│  (Judul + Isi Teks / Postingan / URL)        │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│         NLP PREPROCESSING MODULE             │
│  • Lowercase, hapus URL/emoji               │
│  • Normalisasi slang (80+ mapping)          │
│  • Ekstraksi 14 fitur linguistik            │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│         LLM CLAIM EXTRACTION (Gemini)        │
│  1 API call per article (shared)             │
│  Output: daftar klaim terstruktur (1-N)      │
│  + claim_type: causal/factual/statistical/   │
│    attribution/opinion                       │
└──────────────────┬──────────────────────────┘
                   ↓
          ┌────────┴────────┐
          ↓                 ↓
┌───────────────────┐ ┌──────────────────────┐
│  LSTM (FULL TEXT)  │ │ EVIDENCE RETRIEVAL   │
│                   │ │                      │
│  Input: full       │ │  Tier 1: Google FC   │
│  article text      │ │  Tier 2: ChromaDB    │
│  Output:           │ │  (BM25 + embedding)  │
│  hoax_proba [0,1]  │ │  Tier 3: RSS feeds   │
│  (binary: hoax)    │ │  Tier 4: Wikipedia   │
│                   │ │                      │
└────────┬──────────┘ │  Rule-based query gen │
         ↓            │  (no LLM, saves 30%)  │
                      └──────────┬───────────┘
                                 ↓
                  ┌──────────────────────────────┐
                  │  LLM EVIDENCE JUDGE (Gemini) │
                  │  1 API call per claim        │
                  │  Input: claim + top-3 evid.  │
                  │  Output: Supported/Refuted/  │
                  │    NEI + confidence + reason │
                  └──────────┬───────────────────┘
                             ↓
                  ┌──────────────────────────────┐
                  │  CONFIDENCE FUSION ENGINE    │
                  │  Regime-based (bukan linear): │
                  │  • Strong evidence → trust   │
                  │    LLM heavily (50%)         │
                  │  • Weak evidence → lean LSTM │
                  │    (55%) + low confidence    │
                  │  • No evidence → LSTM only,  │
                  │    pull toward NEI (0.5)     │
                  │  Evidence quality =          │
                  │  confidence multiplier       │
                  │                              │
                  │  ⚠️ "Tidak Cukup Bukti"       │
                  │  muncul saat evidence = 0,   │
                  │  bukan dari kelas LSTM       │
                  └──────────┬───────────────────┘
                             ↓
                  ┌──────────────────────────────┐
                  │  ARTICLE-LEVEL AGGREGATION    │
                  │  Weighted by:                 │
                  │  • Claim importance           │
                  │  • Claim type priority        │
                  │  • Proportion hoax vs valid   │
                  │  Bukan "1 hoax = artikel hoax"│
                  └──────────┬───────────────────┘
                             ↓
                  ┌──────────────────────────────┐
                  │  FINAL OUTPUT                 │
                  │  { verdict, confidence,       │
                  │    claims[], evidence[],      │
                  │    reasoning, stats }         │
                  └──────────────────────────────┘
```

---

## 3. Detail Per Modul

### 3.1 NLP Preprocessing Module

**Tujuan:** Membersihkan dan menormalisasi teks Bahasa Indonesia sebelum diproses lebih lanjut.

#### 3.1.1 Text Cleaning Pipeline

```
Raw Text
  ↓
1. Case folding → lowercase
2. Hapus URL (http://, https://, www.)
3. Hapus mention (@username) dan hashtag (#...)
4. Hapus emoji berlebihan (sisakan maks. 1 per kalimat)
5. Hapus karakter khusus yang tidak informatif
6. Normalisasi tanda baca berlebihan (!!!, ???) → (!, ?)
7. Hapus spasi ganda dan trim
```

#### 3.1.2 Normalisasi Slang / Bahasa Tidak Baku

Mapping menggunakan dictionary lookup dengan 80+ entri (tanpa duplicate keys):

| Slang | Normal |
|---|---|
| gk, ga, gak, ngga, nggak | tidak |
| bgt | banget |
| yg | yang |
| dg | dengan |
| krn | karena |
| jgn | jangan |
| udh, udah | sudah |
| klo, klu | kalau |
| sm | sama |
| dr | dari |
| sy, gua, gw | saya |
| bkn | bukan |
| tp | tapi |
| dpt | dapat |
| nge- | me- (prefix) |
| mager | malas bergerak |
| santuy | santai |
| wkwk | tertawa |

Library yang dipakai: **Sastrawi** (stemmer Bahasa Indonesia).

#### 3.1.3 Metadata Feature Extraction (14 Fitur)

| Fitur | Tipe | Contoh |
|---|---|---|
| `text_length` | int | jumlah karakter |
| `word_count` | int | jumlah kata |
| `sentence_count` | int | jumlah kalimat |
| `avg_word_length` | float | rata-rata panjang kata |
| `caps_ratio` | float | proporsi huruf kapital |
| `exclamation_count` | int | jumlah tanda seru |
| `question_count` | int | jumlah tanda tanya |
| `provocative_word_count` | int | kata-kata provokatif |
| `clickbait_score` | float | skor berdasarkan pola clickbait |
| `sentiment_score` | float | dari sentiment analyzer (-1 s/d +1) |
| `has_source_mention` | bool | ada nama institusi sumber |
| `has_date_mention` | bool | ada tanggal/waktu spesifik |
| `has_data_mention` | bool | ada angka/statistik |
| `urgency_words` | int | kata-kata mendesak |

Fitur ini di-normalisasi (MinMax) lalu concat ke branch LSTM untuk feature fusion.

---

### 3.2 LLM Claim Extraction Module (Gemini)

**Tujuan:** Mengekstrak klaim-klaim faktual yang dapat diverifikasi dari teks.

#### 3.2.1 Input & Output

**Input:** Teks yang sudah dibersihkan dari Preprocessing Module.

**Output:** JSON terstruktur:

```json
{
  "claims": [
    {
      "claim_id": 1,
      "claim_text": "Vaksin menyebabkan gagal ginjal massal",
      "claim_type": "causal",
      "original_sentence": "Vaksin yang diberikan pemerintah menyebabkan gagal ginjal massal di Indonesia.",
      "entities": ["vaksin", "gagal ginjal", "Indonesia"],
      "importance": 1.0
    }
  ]
}
```

#### 3.2.2 Claim Types

| Tipe | Keterangan | Weight | Contoh |
|---|---|---|---|
| `causal` | Klaim sebab-akibat | 1.0 | "A menyebabkan B" |
| `factual` | Klaim fakta | 1.0 | "Terjadi X di tempat Y" |
| `statistical` | Klaim statistik/angka | 0.8 | "80% orang mengalami X" |
| `attribution` | Klaim atribusi | 0.6 | "Menurut WHO, X benar" |
| `opinion` | Opini / bukan fakta | 0.0 | "Menurut saya X buruk" |

**Note:** Klaim tipe `opinion` (weight 0.0) tidak masuk ke pipeline verifikasi — langsung dilewati di aggregation.

#### 3.2.3 Cost Optimization

- Claim extraction: **1 API call** per article (shared untuk semua klaim)
- Query generation: **0 API call** (rule-based, menghemat 30%)
- Evidence judge: **1 API call per claim**
- Total: ~4 LLM API calls per article (rata-rata 3 klaim)

#### 3.2.4 Fallback Mechanism

Kalau Gemini API gagal/unavailable → fallback ke rule-based claim extraction:
- Split by sentence
- Filter sentences dengan pola klaim (kata kunci: menyebabkan, menurut, dilaporkan, dll)
- Assign claim_type berdasarkan keyword matching

---

### 3.3 LSTM Classifier Module (BINARY)

**Tujuan:** Binary classification — apakah gaya bahasa artikel mirip hoaks atau tidak.

> **Kenapa binary, bukan 3-class?**
>
> "Tidak Cukup Bukti" / "uncertain" **bukan pola bahasa** — itu adalah **keadaan evidence**.
> Artinya: klaim diajukan, tapi sumber belum mengklarifikasi. Ini tidak bisa dipelajari dari gaya tulisan.
>
> Contoh:
> - Hoax → caps lock berlebihan, emoji banyak, kata "SEBARKAN!", bahasa provokatif
> - Valid → jurnalistik netral, ada sumber, ada tanggal/tempat
> - Uncertain → **tidak ada pola bahasa khusus** — ini terjadi saat evidence retrieval tidak menemukan bukti
>
> Jadi LSTM cuma perlu belajar membedakan **2 pola**: hoax vs valid.
> "Tidak Cukup Bukti" muncul otomatis di fusion engine saat evidence_quality = 0.

#### 3.3.1 Model Arsitektur — BiLSTM

```
Input: full article text (tokenized, max_len=200)
  ↓
Embedding Layer (vocab_size=20000, dim=128, mask_zero=True)
  ↓
Dropout (0.3)
  ↓
Bidirectional LSTM (units=64, return_sequences=True, dropout=0.3)
  ↓
Bidirectional LSTM (units=32, return_sequences=False)
  ↓
Dense (32, ReLU)
  ↓
Dropout (0.3)
  ↓
Dense (2, Softmax) → [hoax, valid]
```

#### 3.3.2 Feature Fusion Branch (Opsional)

```
LSTM text output     → 64-dim vector
                                CONCAT → Dense(32) → Dense(2, Softmax)
Linguistic features  → 14-dim vector (normalized)
```

#### 3.3.3 Training Setup

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Batch size | 64 |
| Epochs | 20 (early stopping patience=5) |
| Max sequence length | 200 |
| Loss | Sparse Categorical Crossentropy |
| Class weights | Otomatis dari sklearn |
| Framework | TensorFlow/Keras |
| Training location | **Google Colab (T4 GPU, gratis)** |

#### 3.3.4 Dataset Strategy — Binary (Hoax vs Valid)

| Label | Minimum | Ideal | Sumber Utama |
|---|---|---|---|
| **hoax** | 3,000 | 10,000+ | TurnBackHoax, MAFINDO, Kominfo |
| **valid** | 2,000 | 5,000+ | Kompas, Tempo, BMKG, BPOM |
| **TOTAL** | **5,000** | **15,000+** | |

> **Tidak perlu dataset "uncertain" untuk training LSTM.**
> "Tidak Cukup Bukti" muncul dari fusion engine saat evidence tidak ditemukan.

**Data Split:**
```
Training:   70%  (stratified by class AND source)
Validation: 15%
Test:       15%  (time-based: data terbaru sebagai test)
```

#### 3.3.5 Output per Artikel

```json
{
  "lstm_hoax_proba": 0.82,
  "lstm_valid_proba": 0.18
}
```

`lstm_hoax_proba` inilah yang masuk ke fusion engine sebagai `lstm_hoax` parameter.

#### 3.3.6 File Deployment (Yang Dibutuhkan Sistem)

Saat training di Colab, banyak file dihasilkan. Yang **dibutuhkan untuk deployment** hanya 3:

| File | Dipakai? | Fungsi |
|---|---|---|
| `lstm_model.keras` | ✅ **YA** | Model weights — otak LSTM |
| `tokenizer.pkl` | ✅ **YA** | Convert text → angka sequences (harus sama dengan saat training) |
| `label_map.json` | ✅ **YA** | Mapping index → label name |
| `training_history.png` | ❌ | Visualisasi only, untuk paper |
| `confusion_matrix.png` | ❌ | Evaluasi only, untuk paper |
| `metrics_report.json` | ❌ | Evaluasi only, untuk paper |
| `checkpoint_epoch_XX.keras` | ❌ | Intermediate, sudah diganti final model |

File yang ditaruh di `models/lstm/`:
```
models/lstm/
├── lstm_model.keras
├── tokenizer.pkl
└── label_map.json
```

---

### 3.4 Evidence Retrieval Module

**Tujuan:** Mencari bukti yang relevan untuk setiap klaim dari sumber kredibel.

#### 3.4.1 Tiered Source Architecture

| Prioritas | Sumber | Strategi | Biaya |
|---|---|---|---|
| Tier 1 | Google Fact Check API | Direct API call, cache 7 hari | Gratis |
| Tier 2 | Local ChromaDB | BM25 + embedding, pre-scraped data | Gratis |
| Tier 3 | RSS feeds (Kemenkes, BMKG) | Scheduled crawl → indexed | Gratis |
| Tier 4 | Wikipedia API | Fallback general knowledge | Gratis |

#### 3.4.2 Hybrid Retrieval Flow

```
Claim Text
    ↓
Rule-based Query Generation (NO LLM — saves 30%)
    ↓
┌──────────┬──────────┬──────────┐
│ Google   │ ChromaDB │ Wikipedia│
│ FC API   │ (BM25+   │ (T4)     │
│ (T1)     │ emb) (T2)│          │
└────┬─────┴────┬─────┴────┬─────┘
     ↓          ↓          ↓
     Results Merger & Reranker
     • Deduplicate by URL
     • Hybrid score: 0.4*BM25 + 0.6*embedding
     • Source credibility boost
     • Top-3 return
     ↓
SQLite Cache (7-day TTL)
```

#### 3.4.3 Embedding Model (Gratis, Local)

```
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```
~300MB, download sekali saja saat pertama kali run. Runs on CPU.

#### 3.4.4 Skema Dokumen Evidence

```json
{
  "id": "evidence_001",
  "source": "Kemenkes RI",
  "source_tier": 1,
  "url": "https://kemkes.go.id/...",
  "title": "Fakta tentang Gagal Ginjal",
  "text": "Tidak ada bukti ilmiah bahwa konsumsi matcha...",
  "category": "health",
  "date_published": "2025-03-15"
}
```

#### 3.4.5 Scoring

Hybrid score = 0.4 × BM25_score + 0.6 × semantic_score
Final score = hybrid_score × credibility_weight

Credibility weights:
- Tier 1: 1.0 (Kemenkes, WHO, BMKG, BPOM)
- Tier 2: 0.9 (Kompas, Tempo, Detik)
- Tier 3: 0.75 (MAFINDO/TurnBackHoax)
- Tier 4: 0.4 (Blog/forum/media kecil)

---

### 3.5 LLM Evidence Judge Module (Gemini)

**Tujuan:** Menganalisis klaim bersama evidence dan memberikan verdict berbasis bukti.

#### 3.5.1 Input & Output

**Input:**
```json
{
  "claim": "Minum matcha setiap hari menyebabkan gagal ginjal",
  "evidence": [
    { "source": "Kemenkes RI", "text": "..." },
    { "source": "TurnBackHoax", "text": "..." }
  ]
}
```

**Output:**
```json
{
  "claim_id": 1,
  "verdict": "Refuted",
  "confidence": 0.88,
  "reasoning": "Klaim dibantah oleh Kementerian Kesehatan RI...",
  "evidence_used": ["Kemenkes RI", "TurnBackHoax"]
}
```

#### 3.5.2 Verdict Types

| Verdict | Arti |
|---|---|
| **Supported** | Evidence kuat mendukung klaim → klaim VALID |
| **Refuted** | Evidence kuat membantah klaim → klaim HOAX |
| **NotEnoughEvidence** | Evidence tidak cukup/ambigu → NEI |

#### 3.5.3 Constraint

LLM **hanya boleh menggunakan evidence yang diberikan**, tidak boleh mengarang sumber baru. Prompt constraint enforced.

---

### 3.6 Confidence Fusion Engine (REVISED — Critical Fix)

**Tujuan:** Menggabungkan semua sinyal menjadi keputusan final per klaim.

#### 3.6.1 Masalah di Formula Lama (v1)

Formula lama menambahkan evidence_relevance, source_credibility, recency sebagai skor hoax independen → **arahless**, mendorong skor ke atas meskipun evidence menunjukkan "Supported".

Contoh bug:
```
lstm = 0.85 (curiga hoax)
llm = 0.00 (Supported — klaim valid)
relevance = 0.80 (evidence sangat relevan)
credibility = 0.40 (sumber kurang kredibel)

Formula lama: 0.30(0.85) + 0.40(0.00) + 0.15(0.80) + 0.10(0.40) + 0.05(recency)
            = 0.255 + 0.000 + 0.120 + 0.040 + ...
            = 0.445 → "Tidak Cukup Bukti"

Padahal LLM sudah bilang Supported! Tapi evidence_relevance mendorong skor ke atas.
```

#### 3.6.2 Formula Baru — Evidence Quality = Confidence Multiplier

```python
# Step 1: LLM → directional signal
# Refuted → +confidence (hoax direction), Supported → -confidence (valid direction)
if llm_verdict == "Refuted":
    llm_signal = llm_confidence       # positif → hoax
elif llm_verdict == "Supported":
    llm_signal = -llm_confidence      # negatif → valid
else:  # NEI
    llm_signal = 0.0                  # netral

llm_hoax_normalized = (llm_signal + 1.0) / 2.0  # [-1,+1] → [0,1]
```

```python
# Step 2: Evidence Quality (bukan skor hoax!)
evidence_quality = 0.50 * relevance + 0.30 * credibility + 0.20 * recency
# [0, 1] — hanya mengukur KUALITAS bukti, bukan arah hoax
```

```python
# Step 3: Regime-based Fusion
if evidence_quality >= 0.50:
    # STRONG EVIDENCE — trust LLM + evidence
    final_hoax = 0.25 * lstm + 0.50 * llm_hoax_normalized + 0.10 * linguistic
    confidence = evidence_quality * (1.0 - conflict * 0.3)

elif evidence_quality > 0:
    # WEAK EVIDENCE — lean on LSTM, low confidence
    final_hoax = 0.55 * lstm + 0.10 * llm_hoax_normalized + 0.25 * linguistic
    confidence = evidence_quality * 0.5

else:
    # NO EVIDENCE — LSTM only, pull toward NEI (0.5)
    final_hoax = 0.69 * lstm + 0.31 * linguistic
    # Pull toward 0.5 if LSTM uncertain
    confidence = 0.3 + 0.2 * (1.0 - lstm_uncertainty)
```

```python
# Step 4: Verdict mapping
if final_hoax > 0.70: verdict = "Hoax"
elif final_hoax < 0.30: verdict = "Tidak Hoax"
else: verdict = "Tidak Cukup Bukti"
```

#### 3.6.3 Bagaimana "Tidak Cukup Bukti" Muncul (Binary LSTM)

Karena LSTM cuma binary (hoax vs valid), "Tidak Cukup Bukti" muncul dari **regime fusion**, bukan dari kelas LSTM:

```
Skenario A: Evidence kuat → verdict dari LLM + evidence (bukan LSTM)
  → Bisa "Hoax" atau "Tidak Hoax"

Skenario B: Evidence lemah → verdict condong ke LSTM, confidence rendah
  → Biasanya "Tidak Cukup Bukti" karena confidence rendah

Skenario C: Evidence TIDAK ADA (evidence_quality = 0)
  → Skor ditarik ke 0.5 → verdict otomatis "Tidak Cukup Bukti"
  → Ini adalah kasus "uncertain" yang sesungguhnya
```

#### 3.6.4 Key Changes v1 → v3

| Issue | Sebelum (v1) | Sesudah (v3) |
|---|---|---|
| Evidence scores | + 0.15*relevance + 0.10*credibility | Digabung jadi evidence_quality → confidence multiplier |
| Source credibility | Dihitung 2x (LLM + Fusion) | Hanya di evidence_quality, tidak additive di fusion |
| NEI handling | Selalu mapped ke 0.5 | "no_results" = netral, "ambiguous" = weak signal |
| Weights | Static | Adaptive: strong/weak/no evidence regime |
| LSTM classes | 3-class (hoax/valid/uncertain) | **Binary** (hoax/valid) — uncertain dari fusion |
| "Tidak Cukup Bukti" | Kelas LSTM | **Regime fusion result** saat evidence = 0 |

---

### 3.7 Article-Level Aggregation Module (REVISED)

**Tujuan:** Menggabungkan verdict per-klaim menjadi verdict untuk seluruh artikel.

#### 3.7.1 Masalah di v1

"Jika ada ≥1 klaim Hoax → Artikel = Hoax" → terlalu agresif. Artikel valid dengan 1 kalimat opinion yang salah-classified jadi "Hoax" padahal intinya valid.

#### 3.7.2 Rules Baru — Weighted Aggregation

1. Opinion claims di-skip (weight 0.0)
2. Setiap klaim diberi weight = importance × claim_type_weight
3. Weighted average hoax score
4. Verdict berdasarkan kombinasi:
   - High-confidence hoax claim (conf ≥ 0.70, weight ≥ 0.5) + hoax > valid → Hoax
   - Semua verifiable = Tidak Hoax → Tidak Hoax
   - Campuran → Tidak Cukup Bukti

#### 3.7.3 Claim Type Weights

| Type | Weight | Alasan |
|---|---|---|
| factual | 1.0 | Klaim langsung paling penting |
| causal | 1.0 | Sebab-akibat kritis untuk diverifikasi |
| statistical | 0.8 | Klaim angka |
| attribution | 0.6 | Atribusi (kurang kritis) |
| opinion | 0.0 | Opini tidak diverifikasi |

#### 3.7.4 Output Final Sistem

```json
{
  "verdict": "Hoax",
  "confidence": 0.72,
  "avg_hoax_score": 0.78,
  "summary": "Artikel mengandung 1 klaim yang dibantah oleh sumber kredibel...",
  "claims": [
    {
      "claim_text": "Matcha menyebabkan gagal ginjal",
      "claim_type": "causal",
      "verdict": "Hoax",
      "confidence": 0.86,
      "mode": "strong_evidence",
      "reasoning": "...",
      "evidence_sources": ["Kemenkes RI", "TurnBackHoax"]
    }
  ],
  "claim_stats": {
    "total_claims": 3,
    "verifiable_claims": 2,
    "hoax_claims": 1,
    "valid_claims": 1,
    "nei_claims": 0
  },
  "processing_time_ms": 3450
}
```

---

## 4. Dataset & Training

### 4.1 Dataset Utama (Bahasa Indonesia) — Binary

| Dataset | Sumber | Estimasi | Label | Cara Akses |
|---|---|---|---|---|
| **TurnBackHoax** | MAFINDO | ~12,000-15,000 | Hoax | Scraping (1 req/2s) |
| **CekFakta** | Tempo | ~2,000-3,000 | Hoax/Valid | Scraping |
| **Kominfo Hoax** | Kominfo | ~5,000+ | Hoax | Scraping |
| **ISHOX** | Kaggle/Mendeley | ~3,000-5,000 | Hoax/Non-hoax | Download |
| **Media Kredibel** | Kompas/Tempo | ~3,000-5,000 | Valid | Scraping section non-hoax |

### 4.2 Estimated Totals After Collection

| Kelas | Estimasi | Notes |
|---|---|---|
| Hoax | ~15,000-20,000 | TurnBackHoax + Kominfo + ISHOX |
| Valid | ~8,000-12,000 | News valid + cekfakta clarified |

> **Tidak ada kelas "uncertain" di dataset LSTM.**
> "Tidak Cukup Bukti" adalah output fusion engine, bukan kelas klasifikasi.

### 4.3 Data Split

```
Training:   70%  (stratified by class AND source)
Validation: 15%
Test:       15%  (time-based: data terbaru sebagai test)
```

### 4.4 Training Location

**Google Colab (Recommended)** — T4 GPU gratis:
- Training time: **2-5 menit** (vs 10-30 menit di CPU lokal)
- Tidak perlu install TensorFlow di laptop
- RAM 12GB+ tersedia
- Semua dependency sudah ada
- Notebook: `notebooks/colab_lstm_training.ipynb`

Setelah training selesai di Colab → download model (lstm_model.zip) → extract ke `models/lstm/` → jalankan API + UI di laptop lokal.

### 4.5 Class Imbalance Handling

```
• Class weights dalam loss function (otomatis dari sklearn)
• Data augmentation: synonym replacement, back-translation
• Oversampling minority class jika diperlukan
```

---

## 5. Evaluasi

### 5.1 Model Classification Metrics

| Metrik | Target | Alasan |
|---|---|---|
| **Accuracy** | > 85% | Overall correctness |
| **Macro-F1** | > 0.80 | Penting karena imbalance |
| **Recall (Hoax)** | > 0.85 | Jangan sampai hoax terlewat |
| **Precision (Hoax)** | > 0.80 | Minimalkan false positive |
| **ROC-AUC** | > 0.90 | Separability antar kelas |

### 5.2 Evidence Retrieval Metrics

| Metrik | Target |
|---|---|
| **Precision@3** | > 0.70 |
| **NDCG@5** | > 0.75 |
| **Mean Reciprocal Rank** | > 0.60 |

### 5.3 End-to-End Metrics

| Metrik | Target |
|---|---|
| **False Positive Rate** | < 10% |
| **False Negative Rate** | < 15% |
| **Average processing time** | < 10 detik/article |

---

## 6. Tech Stack

### 6.1 Core (Semua Modul)

```
Python 3.10+
google-generativeai   (LLM Gemini)
fastapi + uvicorn     (Backend API)
streamlit             (Demo UI)
pydantic              (Request/response validation)
python-dotenv         (.env file loading)
pyyaml                (Config files)
```

### 6.2 ML & NLP

```
tensorflow>=2.15      (LSTM training — di Colab)
nltk + sastrawi       (NLP Bahasa Indonesia)
pandas, numpy         (Data processing)
scikit-learn          (Evaluation metrics, class weights)
```

### 6.3 Evidence Retrieval

```
chromadb>=0.4         (Vector database)
sentence-transformers (Embedding model — multilingual)
rank_bm25             (Keyword search)
wikipedia-api         (Wikipedia fallback)
requests, bs4, lxml   (Web scraping)
```

### 6.4 API Cost Estimation

| Scenario | Usage | Cost |
|---|---|---|
| Demo (10 articles/day) | 40 calls/day | **$0** (free tier) |
| Testing (50 articles/day) | 200 calls/day | **$0** (free tier) |
| Heavy (200+/day) | 800+ calls/day | ~$1-3/bulan |

**Free tier Gemini 2.0 Flash:** ~1,500 requests/hari — cukup untuk demo.

---

## 7. Struktur Folder Project

```
FAKTA/
├── Arsitektur.md                  # Dokumen ini
├── PANDUAN_LENGKAP.md             # Step-by-step guide
├── README.md                      # Project overview
├── requirements.txt               # Dependencies
├── .env.example                   # Template env vars
├── .env                           # Actual env (gitignored)
├── .gitignore
│
├── configs/
│   ├── fusion_config.yaml         # Fusion weights & thresholds
│   ├── source_tiers.yaml          # 5-tier source credibility
│   └── lstm_config.yaml           # LSTM hyperparameters
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── cleaning.py            # Text cleaning pipeline
│   │   ├── slang_normalizer.py    # 80+ slang → formal ID
│   │   └── feature_extractor.py   # 14 linguistic features
│   ├── claim_extraction/
│   │   └── gemini_extractor.py    # LLM claim extraction + fallback
│   ├── classifier/
│   │   ├── lstm_model.py          # BiLSTM model builder
│   │   ├── train_lstm.py          # Training pipeline
│   │   └── predict_lstm.py        # CLI inference
│   ├── evidence/
│   │   ├── retriever.py           # Hybrid BM25 + ChromaDB
│   │   ├── factcheck_api.py       # Google Fact Check API
│   │   ├── wikipedia_fallback.py  # Tier 4 fallback
│   │   ├── cache.py               # SQLite cache + RateLimiter
│   │   ├── indexer.py             # Batch ingestion ke ChromaDB
│   │   └── source_scoring.py      # Credibility & recency scoring
│   ├── judge/
│   │   └── gemini_evidence_judge.py  # LLM evidence judge
│   ├── fusion/
│   │   ├── confidence_fusion.py   # REVISED fusion engine ⭐
│   │   └── aggregation.py         # Weighted article aggregation
│   ├── data/
│   │   └── collect.py             # TurnBackHoax scraper
│   └── api/
│       ├── main.py                # FastAPI backend + full pipeline
│       └── schemas.py             # Pydantic schemas
│
├── app/
│   └── streamlit_app.py           # Demo UI interaktif
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_lstm_training.ipynb
│   ├── 03_evidence_retrieval_test.ipynb
│   ├── 04_end_to_end_evaluation.ipynb
│   └── colab_lstm_training.ipynb  # ⭐ Google Colab notebook
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── training/                  # CSV files for LSTM (hoax + valid only)
│   ├── evaluation/
│   └── evidence/                  # ChromaDB + BM25 index
│
└── models/
    └── lstm/                      # lstm_model.keras, tokenizer.pkl, label_map.json
```

---

## 8. API Specification

### 8.1 Endpoint Utama

```
POST /check
```

**Request:**
```json
{
  "text": "VIRAL!!! Matcha menyebabkan gagal ginjal!! Sebarkan!",
  "title": "Bahaya Matcha"
}
```

**Response:**
```json
{
  "verdict": "Hoax",
  "confidence": 0.72,
  "avg_hoax_score": 0.78,
  "summary": "Artikel mengandung klaim yang dibantah...",
  "claims": [...],
  "claim_stats": {
    "total_claims": 2,
    "verifiable_claims": 2,
    "hoax_claims": 1,
    "valid_claims": 1,
    "nei_claims": 0
  },
  "processing_time_ms": 3420
}
```

### 8.2 Endpoint Lainnya

```
GET  /health          → Health check + component status
GET  /stats           → System stats (cache, retriever)
POST /feedback        → Collect human feedback for tuning
```

---

## 9. Design Decisions & Justifikasi

| Keputusan | Alasan |
|---|---|
| **Hybrid bukan single model** | LSTM lemah di evidence, LLM bisa halusinasi — kombinasi lebih robust |
| **LSTM > IndoBERT** | Lebih ringan, cocok untuk demo/akademis, cukup akurat dengan data besar |
| **Binary LSTM (bukan 3-class)** | "Uncertain" bukan pola bahasa — itu keadaan evidence. Lebih clean secara akademis |
| **Claim-based bukan article-based** | Verifikasi klaim individual lebih akurat daripada classify seluruh artikel |
| **Evidence quality = confidence multiplier** | Fix bug v1: evidence tidak boleh jadi skor hoax independen |
| **Regime-based fusion** | Weight adaptive berdasarkan kualitas evidence — bukan static |
| **Weighted aggregation** | "1 hoax = artikel hoax" terlalu agresif — perlu weight by importance |
| **Rule-based query generation** | Hemat 30% API calls — tidak perlu LLM untuk generate search query |
| **Training di Colab** | GPU gratis, 5-10x lebih cepat dari CPU lokal |
| **LLM dikunci prompt constraint** | Mencegah halusinasi — LLM hanya boleh pakai evidence yang diberikan |

---

## 10. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Dataset tidak cukup | Model underfit | Augmentasi data, scraping lebih banyak, Colab untuk train cepat |
| Evidence database kosong | Sistem tidak bisa verifikasi | Mulai dari TurnBackHoax, expand bertahap |
| LLM rate limit / mahal | Biaya operasional | Free tier cukup untuk demo, caching, rule-based query gen |
| False positive tinggi | Berita valid salah label | Threshold tuning, feedback loop, weighted aggregation |
| Bahasa slang tidak ter-cover | Preprocessing gagal | 80+ slang dictionary, iteratif perbaiki |
| Hoax baru belum di database | Evidence retrieval gagal | Label "Tidak Cukup Bukti", bukan langsung "Hoax" |
| API Gemini down | Claim extraction/judge gagal | Fallback rule-based claim extraction, LSTM-only mode |

---

## 11. Roadmap Implementasi

### Phase 1: Foundation (Weeks 1-2) — MVP CORE

| Task | Est | Why |
|---|---|---|
| Data collection pipeline | 3 hari | No data = no LSTM |
| Text preprocessing module | 2 hari | Dipakai semua downstream |
| LSTM training (binary) | 4 hari | Core signal |
| ChromaDB + BM25 setup | 3 hari | Evidence backbone |

### Phase 2: LLM Integration (Weeks 3-4) — HYBRID SYSTEM

| Task | Est | Why |
|---|---|---|
| Gemini API integration | 2 hari | Hybrid system enabled |
| Fusion engine (revised) | 2 hari | Critical bug fix |
| Evidence caching | 1 hari | Prevents redundant API |
| Google Fact Check API | 1 hari | Free external evidence |

### Phase 3: Refinement (Weeks 5-6) — QUALITY

| Task | Est | Why |
|---|---|---|
| Article weighted aggregation | 2 hari | Replaces broken rule |
| Evidence conflict detection | 1 hari | Handle contradictory evidence |
| LSTM linguistic feature fusion | 2 hari | Improve LSTM accuracy |

### Phase 4: UI & Demo (Weeks 7-8) — POLISH

| Task | Est | Why |
|---|---|---|
| Streamlit UI | 3 hari | Demo untuk thesis |
| FastAPI backend | 2 hari | Clean API layer |
| Feedback endpoint | 1 hari | Show extensibility |

---

## 12. Kalimat untuk Paper / Skripsi

> "Penelitian ini mengembangkan sistem fact-checking berbahasa Indonesia berbasis hybrid LSTM dan LLM-assisted evidence reasoning. LSTM dengan arsitektur binary (hoax vs valid) digunakan untuk mempelajari pola linguistik dari dataset berlabel, sedangkan LLM (Gemini) digunakan untuk ekstraksi klaim dan reasoning berbasis evidence. Verifikasi klaim dilakukan melalui multi-source evidence retrieval dari Google Fact Check API, local vector database (ChromaDB + BM25), serta Wikipedia sebagai fallback. Keputusan akhir ditentukan melalui confidence fusion adaptif yang menggabungkan probabilitas LSTM, verdict LLM, dan kualitas evidence — di mana evidence quality berfungsi sebagai confidence multiplier, bukan skor hoax independen. Label 'Tidak Cukup Bukti' dihasilkan secara otomatis oleh fusion engine saat evidence tidak ditemukan (evidence_quality = 0), bukan sebagai kelas klasifikasi terpisah. Sistem ini dirancang untuk mengatasi keterbatasan klasifikasi teks tradisional dengan menambahkan verifikasi berbasis bukti pada setiap klaim, dan menghasilkan verdict yang lebih interpretable melalui regime-based fusion."

---

## 13. Quick Commands Reference

```bash
# === SETUP (sekali saja) ===
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# === DATA ===
python src/data/collect.py              # Scrape TurnBackHoax

# === TRAINING (di Google Colab) ===
# Upload notebooks/colab_lstm_training.ipynb → Colab
# Select T4 GPU → Run all → Download model → Extract to models/lstm/

# === EVIDENCE (sekali) ===
python src/evidence/indexer.py           # Index evidence to ChromaDB

# === JALANKAN ===
python src/api/main.py                   # Terminal 1: API server
streamlit run app/streamlit_app.py       # Terminal 2: Demo UI

# === TESTING ===
python src/fusion/confidence_fusion.py   # Test fusion engine
python src/preprocessing/cleaning.py     # Test preprocessing
python src/classifier/predict_lstm.py "text here"  # Test LSTM
python src/evidence/retriever.py         # Test retrieval
```
