# Unlox - Data-Driven Social Engagement Initiative

This is my data science project. The idea is to analyse a YouTube channel's performance — pulling video stats, computing a viral score, running sentiment analysis on comments, and showing everything in a dashboard.

I built this using Python. The pipeline runs in stages and the results are shown in a Streamlit dashboard with charts.

---

## Setup

You need Python 3.11. Higher versions like 3.14 don't have pre-built wheels for scipy and some other packages so they fail to install.

**Step 1 — create a virtual environment**
```
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt
```

**Step 2 — add API key (optional)**

Copy `.env.example` to `.env` and fill in your YouTube API key and channel ID.
If you don't have one the project will use generated dummy data automatically, so it still works.

**Step 3 — run the pipeline**
```
venv/bin/python run_pipeline.py
```

**Step 4 — open the dashboard**
```
venv/bin/streamlit run src/dashboard/app.py
```

---

## Folder structure

```
unlox/
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dummy_data.py
│   │   └── youtube_client.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── ab_testing.py
│   │   ├── forecasting.py
│   │   ├── recommender.py
│   │   └── virality_model.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   └── sentiment.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── extract.py
│   │   └── viral_score.py
│   └── utils/
│       ├── __init__.py
│       ├── db.py
│       └── logger.py
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── run_pipeline.py
```

---

## Every file explained

---

### `run_pipeline.py`

The main script that runs the full pipeline. Has 7 stage functions — each one imports and calls a specific pipeline module. Uses `argparse` so you can run a single stage with `--stage` flag or run all stages at once.

**Functions inside:**
- `run_extract()` — calls `run_extraction()` from extract.py
- `run_score()` — calls `run_viral_scoring()` from viral_score.py
- `run_sentiment()` — calls `run_sentiment_pipeline()` from sentiment.py
- `run_ab_test()` — loads the scored CSV and calls `run_all_ab_tests()`
- `run_virality()` — loads the scored CSV and calls `train_model()`
- `run_recommend()` — loads the scored CSV and calls `generate_recommendations()`
- `run_forecast()` — loads the scored CSV and calls `run_forecast()` and `analyse_keyword_trends()`

**Variable inside:**
- `STAGES` — dictionary mapping stage names to their functions, used by the CLI

---

### `requirements.txt`

List of all Python packages. Install with `venv/bin/pip install -r requirements.txt`.

**Packages and why each is needed:**
- `pandas`, `numpy`, `scipy` — data handling and statistical tests
- `google-api-python-client`, `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2` — YouTube Data API v3
- `nltk`, `textblob` — sentiment analysis
- `scikit-learn` — TF-IDF vectorizer, model pipeline, StandardScaler, cross-validation
- `xgboost` — the regression model for predicting viral score
- `statsmodels` — Holt-Winters ETS forecasting fallback
- `prophet` — Facebook Prophet for time-series forecasting
- `streamlit` — dashboard framework
- `plotly` — all charts in the dashboard
- `python-dotenv` — loading `.env` file
- `joblib` — saving and loading the trained model
- `requests` — making HTTP calls to the YouTube API

---

### `.env`

Stores the YouTube API key and channel ID. Not committed to git.

**Variables inside:**
- `YOUTUBE_API_KEY` — your key from Google Cloud Console
- `YOUTUBE_CHANNEL_ID` — the channel ID you want to analyse
- `DB_PATH` — optional, path to the SQLite database file
- `LOG_LEVEL` — optional, logging level (default INFO)

### `.env.example`

Same structure as `.env` but with placeholder values. Safe to commit to git. Shows what variables need to be filled in.

### `.gitignore`

Tells git to ignore `venv/`, `.env`, `__pycache__/`, `.DS_Store`, data files, and model files since those are all generated.

---

### `config/__init__.py`

Empty file. Python needs this to treat the `config` folder as a package so `from config.settings import ...` works in other files.

---

### `config/settings.py`

Central config. Loads `.env` with `load_dotenv()` and exposes all constants used across the project.

**Variables inside:**
- `YOUTUBE_API_KEY` — loaded from `.env`
- `YOUTUBE_CHANNEL_ID` — loaded from `.env`
- `DATA_RAW_DIR` — path to `data/raw/`
- `DATA_PROCESSED_DIR` — path to `data/processed/`
- `MODELS_DIR` — path to `models/`
- `DB_PATH` — path to `data/social_engagement.db`
- `LOG_LEVEL` — logging level string
- `YT_MAX_RESULTS` — max results per YouTube API page (50)
- `YT_COMMENT_PAGES` — how many comment pages to fetch per video (3)
- `RANDOM_STATE` — fixed seed for reproducibility (42)
- `TEST_SIZE` — train/test split ratio (0.2)
- `VIRAL_WEIGHTS` — dict with weights for each metric in the viral score formula

---

### `src/__init__.py`

Empty file. Makes `src` a Python package.

---

### `src/api/__init__.py`

Empty file. Makes `src/api` a package.

---

### `src/api/youtube_client.py`

Connects to the YouTube Data API v3 and fetches all the data we need.

**Variable inside:**
- `BASE_URL` — the YouTube API base URL

**Functions inside:**
- `call_api(endpoint, params)` — makes a GET request to any YouTube API endpoint, attaches the API key, returns the JSON response
- `fetch_channel_stats(channel_id)` — returns subscriber count, total views, video count for a channel
- `fetch_video_ids(channel_id, max_videos)` — returns a list of video IDs from the channel, handles pagination
- `fetch_video_details(video_ids)` — fetches title, view count, likes, comments, duration, tags, thumbnail for each video ID (batches of 50)
- `fetch_comments(video_id, max_pages)` — fetches top-level comments for a video across multiple pages

---

### `src/api/dummy_data.py`

Generates fake but realistic YouTube data when no API key is set. Uses a fixed seed so the data is the same every run.

**Variables inside:**
- `SEED` — random seed value (42)
- `rng` — Python `random.Random` instance with fixed seed
- `np_rng` — NumPy random generator with fixed seed
- `VIDEO_TOPICS` — list of 15 realistic video title topics
- `ALL_TAGS` — list of 15 YouTube tags
- `POSITIVE_COMMENTS` — list of 7 positive comment strings
- `NEGATIVE_COMMENTS` — list of 5 negative comment strings
- `NEUTRAL_COMMENTS` — list of 6 neutral/question comment strings

**Functions inside:**
- `random_id(length)` — generates a random alphanumeric string to use as a fake video/comment ID
- `random_date(days_back)` — generates a random ISO datetime string within the last N days
- `random_duration()` — generates a random YouTube duration string like `PT14M32S`
- `generate_dummy_videos(n)` — generates N fake video rows with log-normal view count distribution
- `generate_dummy_comments(video_ids, comments_per_video)` — generates fake comments for a list of video IDs, mixing positive/negative/neutral in a 6:2:2 ratio

---

### `src/pipeline/__init__.py`

Empty file. Makes `src/pipeline` a package.

---

### `src/pipeline/extract.py`

First pipeline stage. Gets videos and comments either from the YouTube API or dummy data, computes engagement metrics, and saves everything to CSV and SQLite.

**Variables inside:**
- `PLACEHOLDER_KEYS` — set of strings that mean the API key is not configured
- `MIN_VIDEOS` — minimum number of videos to work with (30)
- `MIN_COMMENTS` — minimum number of comments to work with (200)

**Functions inside:**
- `api_is_configured()` — returns True if both API key and channel ID are set and not placeholder values
- `get_videos()` — fetches real videos from YouTube if API is configured, otherwise returns dummy videos. Also pads with dummy videos if the real channel has fewer than 30 videos
- `get_comments(all_video_ids, real_video_ids)` — fetches real comments for up to 10 real videos, then pads with dummy comments to reach MIN_COMMENTS
- `compute_engagement(videos)` — adds `like_ratio`, `comment_ratio`, `engagement_rate` to each video dict
- `save_csv(data, filename)` — saves a list of dicts to a CSV file in `data/raw/`
- `save_to_db(videos, comments)` — initialises the database and bulk inserts videos and comments
- `run_extraction()` — main function, calls all of the above in order, returns two DataFrames

---

### `src/pipeline/viral_score.py`

Second pipeline stage. Reads raw video data and computes a viral score between 0 and 100 for each video using a weighted formula.

**Functions inside:**
- `normalize(series)` — applies min-max normalization to a pandas Series, returns values between 0 and 1. Returns all zeros if min equals max
- `compute_viral_scores(df)` — normalizes each of the 5 metrics (views, likes, comments, like_ratio, engagement_rate), multiplies by the weights from `VIRAL_WEIGHTS`, sums them, and scales to 0-100
- `run_viral_scoring()` — reads `data/raw/videos.csv`, calls `compute_viral_scores()`, saves result to `data/processed/videos_scored.csv`, also updates the viral_score column in the SQLite database

---

### `src/nlp/__init__.py`

Empty file. Makes `src/nlp` a package.

---

### `src/nlp/sentiment.py`

Third pipeline stage. Runs TextBlob sentiment analysis on every comment.

**Variables inside:**
- `POSITIVE_THRESHOLD` — polarity score above this is labelled Positive (0.10)
- `NEGATIVE_THRESHOLD` — polarity score below this is labelled Negative (-0.10)

**Functions inside:**
- `get_label(score)` — takes a polarity float and returns "Positive", "Negative", or "Neutral"
- `analyse_sentiment(df)` — applies TextBlob to the `text` column of a comments DataFrame, adds `sentiment_score`, `sentiment_label`, and `subjectivity_score` columns
- `run_sentiment_pipeline()` — reads `data/raw/comments.csv`, calls `analyse_sentiment()`, saves to `data/processed/comments_sentiment.csv`, and updates the labels in SQLite

---

### `src/ml/__init__.py`

Empty file. Makes `src/ml` a package.

---

### `src/ml/ab_testing.py`

Fourth pipeline stage. Runs Welch's t-tests comparing high vs low values for 4 video attributes.

**Class inside:**
- `ABTestResult` — stores the result of one test: `test_name`, `metric`, `group_a_mean`, `group_b_mean`, `t_stat`, `p_value`, `significant`, `winner`, `effect_size`

**Functions inside:**
- `parse_duration_secs(dur)` — converts YouTube ISO 8601 duration string to total seconds
- `run_ab_test(df, group_col, metric_col, test_name, threshold)` — splits videos by median of `group_col`, runs Welch's t-test on `metric_col`, computes Cohen's d effect size, returns an `ABTestResult`
- `run_all_ab_tests(df)` — runs 4 tests: posting hour vs viral score, video length vs engagement rate, tag count vs viral score, description length vs viral score. Returns a list of `ABTestResult` objects

---

### `src/ml/virality_model.py`

Fifth pipeline stage. Trains an XGBoost model to predict viral score from video features.

**Variables inside:**
- `MODEL_PATH` — path where the trained model is saved (`models/virality_model.joblib`)
- `FEATURES_PATH` — path where the feature names list is saved (`models/virality_features.joblib`)
- `FEATURE_COLS` — list of 16 feature column names used for training

**Functions inside:**
- `parse_duration(dur)` — converts YouTube duration string to seconds
- `build_features(df)` — engineers all features: hour, day_of_week, duration_secs, duration_mins, like_ratio, comment_ratio, engagement_rate, views_per_minute, tag_count, desc_len, title_len, title_word_count, and log transforms of view/like/comment/views_per_minute
- `train_model(df)` — builds a scikit-learn Pipeline (StandardScaler + XGBRegressor), trains it, computes RMSE/MAE/R2 and cross-validation R2, saves model to disk, returns metrics and feature importance
- `predict_viral_score(df)` — loads the saved model and predicts viral scores for a DataFrame
- `predict_single(...)` — predicts the viral score for a single video given its stats, used by the dashboard predictor page

---

### `src/ml/recommender.py`

Sixth pipeline stage. Compares top 25% videos against the rest and generates content strategy recommendations.

**Classes inside:**
- `Recommendation` — holds one recommendation: `category`, `action`, `detail`, `confidence`, `impact`, `evidence`
- `RecommendationReport` — holds the full set of recommendations with metadata. Has a `top(n)` method that returns the N highest confidence recommendations

**Functions inside:**
- `parse_duration_secs(dur)` — converts YouTube duration string to seconds
- `get_impact_label(confidence)` — returns "High", "Medium", or "Low" based on confidence score
- `calc_confidence(lift_pct, n_top)` — computes a confidence score from the percentage lift and sample size
- `posting_time_recs(top, all_videos)` — finds the most common posting hour and day among top videos
- `video_length_recs(top, bottom)` — compares average duration of top vs bottom videos
- `tag_recs(top, all_videos)` — finds optimal tag count and the most common tags in top videos
- `topic_recs(top)` — runs TF-IDF on top video titles to find the best performing keywords
- `engagement_recs(top, all_videos)` — compares engagement rates and like ratios
- `build_weekly_brief(report)` — formats a plain-text summary of the top 5 recommendations
- `generate_recommendations(df)` — splits videos into top/bottom by viral score quartile, calls all the rec functions above, returns a `RecommendationReport`

---

### `src/ml/forecasting.py`

Seventh pipeline stage. Forecasts future view counts and like counts and analyses keyword trends over time.

**Functions inside:**
- `build_timeseries(df, metric)` — aggregates daily totals of a metric into a time series DataFrame with `ds` and `y` columns (Prophet format)
- `forecast_with_prophet(ts, periods, freq)` — fits a Prophet model with yearly and weekly seasonality and returns a forecast DataFrame
- `forecast_with_ets(ts, periods)` — fits a Holt-Winters exponential smoothing model as a fallback when Prophet is not available or has too little data
- `run_forecast(df, metric, periods, freq)` — tries Prophet first, falls back to ETS, handles the case where there are fewer than 3 data points by generating a synthetic series. Returns a dict with the time series, full forecast, future-only rows, and which engine was used
- `analyse_keyword_trends(df, top_n)` — splits videos into two halves by publish date, runs TF-IDF on each half's titles, compares keyword scores and labels each keyword as Rising / Falling / Stable
- `compute_engagement_velocity(df)` — aggregates monthly views/likes/comments and computes month-over-month percentage growth for each

---

### `src/utils/__init__.py`

Empty file. Makes `src/utils` a package.

---

### `src/utils/logger.py`

Simple logging setup using Python's built-in `logging` module.

**Function inside:**
- `get_logger(name)` — creates or returns a logger with the given name. Attaches a StreamHandler that prints `HH:MM:SS [LEVEL] module - message` to the console. Called at the top of every other file as `log = get_logger(__name__)`

---

### `src/utils/db.py`

SQLite database helper. All database operations go through this file.

**Variables inside:**
- `CREATE_VIDEOS_TABLE` — SQL string to create the videos table with all columns
- `CREATE_COMMENTS_TABLE` — SQL string to create the comments table with foreign key to videos

**Functions inside:**
- `get_connection()` — opens a SQLite connection to `DB_PATH` and sets `row_factory` so rows come back as dicts
- `init_db()` — creates the database file and both tables if they don't exist yet
- `execute_query(sql, params)` — runs a SELECT query and returns a list of dicts
- `execute_write(sql, params)` — runs an INSERT/UPDATE/DELETE query and commits
- `bulk_insert(table, rows)` — takes a list of dicts and does a bulk `INSERT OR REPLACE` into the specified table

---

### `src/dashboard/__init__.py`

Empty file. Makes `src/dashboard` a package.

---

### `src/dashboard/app.py`

The Streamlit dashboard. Opens in the browser and shows 9 pages of charts and analysis.

**Pages inside:**
- `Overview` — total videos, views, likes, comments as metric cards. Bar chart of top 10 videos by views
- `Engagement Trends` — line chart of monthly views/likes/comments over time. Month-over-month growth table
- `Viral Score` — histogram of viral score distribution. Scatter plot of views vs engagement coloured by viral score. Table of top 10 videos ranked by viral score
- `Sentiment Analysis` — pie chart of Positive/Neutral/Negative split. Bar chart of sentiment by video. Heatmap of sentiment score per video
- `Best Posting Time` — heatmap of average views by hour and day of week. Bar charts of best posting hour and best day
- `A/B Testing` — table of all 4 test results with p-values, means, and significance. Bar chart comparing group means for each test
- `Virality Predictor` — input form for view count, like count, comment count, duration, tag count, title, hour. Calls `predict_single()` and shows predicted viral score with a gauge chart
- `Recommendations` — cards for each recommendation grouped by category with confidence bar and impact label. Weekly brief text summary
- `Trend Forecast` — line chart of historical data plus 90-day forecast with confidence band. Keyword trend table showing Rising/Falling/Stable keywords

---

## Data folders

| Folder | What is stored |
|--------|---------------|
| `data/raw/videos.csv` | Raw video rows from YouTube API or dummy data. Columns: video_id, title, description, published_at, channel_id, channel_title, view_count, like_count, comment_count, duration, tags, thumbnail_url, like_ratio, comment_ratio, engagement_rate, viral_score, fetched_at |
| `data/raw/comments.csv` | Raw comment rows. Columns: comment_id, video_id, author, text, like_count, published_at, sentiment_label, sentiment_score |
| `data/processed/videos_scored.csv` | Same as videos.csv but with the computed viral_score column filled in |
| `data/processed/comments_sentiment.csv` | Same as comments.csv but with sentiment_label, sentiment_score, and subjectivity_score filled in |
| `data/social_engagement.db` | SQLite database with the same videos and comments data in two tables |
| `models/virality_model.joblib` | Trained XGBoost pipeline saved by joblib after running the virality stage |
| `models/virality_features.joblib` | List of feature column names saved alongside the model so prediction uses the same features |

---

## Running individual stages

```
venv/bin/python run_pipeline.py --stage extract
venv/bin/python run_pipeline.py --stage score
venv/bin/python run_pipeline.py --stage sentiment
venv/bin/python run_pipeline.py --stage ab_test
venv/bin/python run_pipeline.py --stage virality
venv/bin/python run_pipeline.py --stage recommend
venv/bin/python run_pipeline.py --stage forecast
```

---

## Tech stack

- pandas, numpy — data handling
- TextBlob — sentiment analysis
- XGBoost, scikit-learn — machine learning
- scipy — statistical tests
- Facebook Prophet, statsmodels — forecasting
- Streamlit — dashboard
- Plotly — charts
- SQLite (sqlite3) — local database
- python-dotenv — loading .env config
