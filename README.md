# 📊 Data-Driven Social Engagement Initiative

> An end-to-end Data Science & ML ecosystem for YouTube channel analytics —
> built with Python, XGBoost, Prophet, NLP, and Streamlit.

---

## 🚀 Live Demo

```
streamlit run src/dashboard/app.py
→ http://localhost:8501
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Data Sources                              │
│   YouTube Data API v3  ◄──► Realistic Dummy Data (dev)     │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   Extraction Layer  │  src/pipeline/extract.py
          │  CSV  ◄──────────►  SQLite (social_engagement.db)
          └──────────┬──────────┘
                     │
     ┌───────────────┼────────────────────┐
     │               │                    │
     ▼               ▼                    ▼
┌─────────┐   ┌────────────┐   ┌──────────────────┐
│  NLP    │   │  ML Engine │   │  Forecasting     │
│TextBlob │   │  XGBoost   │   │  Prophet + ETS   │
│Sentiment│   │  Virality  │   │  Trend Analysis  │
└────┬────┘   │  A/B Tests │   └────────┬─────────┘
     │        │  Recommender│            │
     │        └──────┬──────┘            │
     └───────────────┼───────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Streamlit Dashboard│  9 interactive pages
          │  Plotly Charts      │
          │  Real-time ML UI    │
          └─────────────────────┘
```

---

## 📁 Project Structure

```
MAJOR project/
├── .env                          ← API keys (never commit)
├── .env.example                  ← Template
├── .gitignore
├── .streamlit/
│   ├── config.toml               ← Dark theme config
│   └── secrets.toml              ← Streamlit Cloud secrets
├── Dockerfile                    ← Container deployment
├── docker-compose.yml
├── requirements.txt
├── run_pipeline.py               ← Master orchestrator
├── setup.sh                      ← One-shot setup
├── config/
│   └── settings.py               ← Central config
├── src/
│   ├── api/
│   │   ├── youtube_client.py     ← YouTube Data API v3
│   │   └── dummy_data.py         ← Dev dummy data generator
│   ├── pipeline/
│   │   ├── extract.py            ← Stage 1: data extraction
│   │   ├── viral_score.py        ← Stage 2: viral scoring
│   │   └── report_generator.py  ← Final HTML report
│   ├── nlp/
│   │   └── sentiment.py          ← TextBlob sentiment classifier
│   ├── ml/
│   │   ├── ab_testing.py         ← Welch's t-test A/B framework
│   │   ├── virality_model.py     ← XGBoost regression model
│   │   ├── recommender.py        ← Strategy recommendation engine
│   │   └── forecasting.py        ← Prophet trend forecaster
│   ├── dashboard/
│   │   └── app.py                ← 9-page Streamlit dashboard
│   └── utils/
│       ├── db.py                 ← SQLite helper
│       └── logger.py             ← Loguru logger
├── data/
│   ├── raw/                      ← videos.csv, comments.csv
│   └── processed/                ← scored + sentiment CSVs
├── models/                       ← Saved ML model (.joblib)
├── logs/                         ← Rotating log files
└── reports/                      ← final_report.html
```

---

## ⚡ Quick Start

### Option 1 — Local (recommended for development)

```bash
# 1. Clone / download the project
cd "MAJOR project"

# 2. One-shot setup
bash setup.sh

# 3. Activate environment
source venv/bin/activate

# 4. (Optional) Add your YouTube API key
#    Edit .env and set YOUTUBE_API_KEY and YOUTUBE_CHANNEL_ID

# 5. Run the full pipeline
python run_pipeline.py

# 6. Launch dashboard
streamlit run src/dashboard/app.py
# → http://localhost:8501
```

### Option 2 — Docker

```bash
# Build
docker build -t social-engagement .

# Run (uses dummy data if no API key)
docker run -p 8501:8501 --env-file .env social-engagement

# Or with docker-compose
docker compose up --build
```

### Option 3 — Streamlit Cloud (free public URL)

1. Push code to GitHub (ensure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path**: `src/dashboard/app.py`
4. Under **Settings → Secrets**, paste your `.env` contents in TOML format:
   ```toml
   YOUTUBE_API_KEY    = "your_key_here"
   YOUTUBE_CHANNEL_ID = "UCxxxxxx"
   ```
5. Click **Deploy** — your app gets a public URL instantly

---

## 🔧 Pipeline Stages

```bash
# Run individual stages
python run_pipeline.py --stage extract    # fetch YouTube data
python run_pipeline.py --stage score      # compute viral scores
python run_pipeline.py --stage sentiment  # NLP sentiment analysis
python run_pipeline.py --stage ab_test    # run A/B statistical tests
python run_pipeline.py --stage virality   # train XGBoost model
python run_pipeline.py --stage recommend  # generate recommendations
python run_pipeline.py --stage forecast   # 90-day Prophet forecast

# Or run everything at once
python run_pipeline.py
```

---

## 📊 Dashboard Pages

| Page | What you see |
|---|---|
| 🏠 Overview | KPIs, top 10 table, views-vs-likes scatter |
| 📈 Engagement Trends | Monthly bar + area charts |
| 🔥 Viral Score | Leaderboard, violin, per-video gauge |
| 💬 Sentiment | Pie, histogram, per-video heatmap |
| ⏰ Best Posting Time | Day-of-week bars + day×hour heatmap |
| 🧪 A/B Testing | Test results + custom test builder |
| 🤖 Virality Predictor | Feature importance + live score form |
| 💡 Recommendations | Confidence-ranked strategy cards + brief |
| 📡 Trend Forecast | Prophet chart + keyword trends + velocity |

---

## 🧠 ML Models

### Virality Prediction (XGBoost Regressor)
- **Target**: `viral_score` (0-100)
- **Features**: 16 engineered (log views, engagement rate, tag count, duration, hour, day, etc.)
- **Performance**: R² = 0.84 | Cross-Val R² = 0.94 ± 0.02

### Sentiment Analysis (TextBlob NLP)
- Polarity score ∈ [-1, +1]
- Labels: Positive (>0.10) | Neutral | Negative (<-0.10)
- Subjectivity score as bonus feature

### Trend Forecasting (Prophet / ETS)
- 90-day view/engagement forecast
- 80% confidence intervals
- Auto-fallback to Holt-Winters ETS if Prophet unavailable

### A/B Testing Framework
- Welch's t-test (unequal variance)
- Cohen's d effect size (small / medium / large)
- Lift % calculation
- Custom test builder in dashboard

---

## 📦 Tech Stack

| Layer | Tools |
|---|---|
| Data | Pandas, NumPy, SQLite |
| API | YouTube Data API v3, requests |
| NLP | TextBlob, scikit-learn TF-IDF |
| ML | XGBoost, scikit-learn, statsmodels |
| Forecasting | Prophet, statsmodels ETS |
| Dashboard | Streamlit, Plotly |
| Viz | Matplotlib, Plotly Express |
| Infra | Docker, Streamlit Cloud |
| Utils | loguru, python-dotenv, joblib |

---

## 📝 Resume Description

> **Data-Driven Social Engagement Initiative** — Built an end-to-end ML analytics
> ecosystem in Python integrating YouTube Data API v3, TextBlob NLP sentiment
> classification, XGBoost virality prediction (R²=0.84), Welch's t-test A/B
> testing framework, TF-IDF content recommendation engine, and Facebook Prophet
> 90-day trend forecasting. Delivered a 9-page interactive Streamlit dashboard
> with real-time ML predictions, statistical analysis, and a custom A/B test
> builder. Deployed via Docker and Streamlit Cloud. Demonstrates full ML
> lifecycle: ingestion → feature engineering → training → evaluation → serving.

---

## 🔮 Future Improvements

- [ ] Instagram Graph API integration (Phase 6)
- [ ] PostgreSQL migration for multi-user deployments
- [ ] BERT-based sentiment (vs TextBlob) for higher accuracy
- [ ] Thumbnail image analysis via OpenCV
- [ ] Email digest of weekly strategy brief
- [ ] Real-time webhook for new video notifications
- [ ] LLM-powered caption generator (OpenAI / Claude API)

---

## 📄 License

MIT — free to use, modify, and deploy.
