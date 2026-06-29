# FAKTA — Fact-Checking AI

Sistem pendeteksi hoaks Bahasa Indonesia berbasis **Hybrid LSTM + LLM + Evidence**.

## Arsitektur

```
User Input → NLP Preprocessing → LLM Claim Extraction
  → LSTM Classifier (parallel) + Evidence Retrieval (parallel)
  → LLM Evidence Judge → Confidence Fusion → Final Verdict
```

## Fitur

- **LSTM Classifier**: Mempelajari pola bahasa hoaks dari dataset berlabel
- **LLM Claim Extraction**: Ekstrak klaim faktual menggunakan Gemini
- **Evidence Retrieval**: Hybrid BM25 + embedding search dari multi-source
- **LLM Evidence Judge**: Reasoning berbasis bukti dari Gemini
- **Confidence Fusion**: Regime-based fusion (strong/weak/no evidence)
- **Article Aggregation**: Weighted aggregation per claim importance

## Instalasi

```bash
pip install -r requirements.txt
```

## Penggunaan

### 1. Data Collection
```bash
python src/data/collect.py
```

### 2. Train LSTM
```bash
python src/classifier/train_lstm.py data/training models/lstm
```

### 3. Index Evidence
```bash
python src/evidence/indexer.py
```

### 4. Run API
```bash
python src/api/main.py
```

### 5. Run Demo UI
```bash
streamlit run app/streamlit_app.py
```

## API Endpoints

| Method | Path | Deskripsi |
|--------|------|-----------|
| GET | `/` | Health check |
| POST | `/check` | Periksa artikel |
| POST | `/feedback` | Kirim feedback |
| GET | `/stats` | Statistik sistem |

## Tech Stack

- **ML**: TensorFlow/Keras (LSTM), scikit-learn
- **LLM**: Google Gemini 2.0 Flash
- **Retrieval**: ChromaDB + BM25 + sentence-transformers
- **API**: FastAPI
- **UI**: Streamlit
- **Data**: Pandas, NLTK, Sastrawi

## Lisensi

MIT License — untuk keperluan akademis.
