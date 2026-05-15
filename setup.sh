#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  setup.sh  –  One-time project environment setup
#  Run: bash setup.sh
# ─────────────────────────────────────────────────────────

set -e  # exit on any error

echo "──────────────────────────────────────────────"
echo "  Data-Driven Social Engagement Initiative"
echo "  Environment Setup"
echo "──────────────────────────────────────────────"

# 1. Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip --quiet

# 3. Install dependencies
echo "[3/5] Installing dependencies..."
pip install -r requirements.txt --quiet

# 4. Download NLTK data
echo "[4/5] Downloading NLTK data..."
python -c "
import nltk
for pkg in ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']:
    nltk.download(pkg, quiet=True)
print('NLTK data ready.')
"

# 5. Copy .env template
echo "[5/5] Setting up .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  → .env created. Edit it and add your YOUTUBE_API_KEY."
else
    echo "  → .env already exists. Skipping."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your YouTube API key (optional)"
echo "  2. Activate venv:   source venv/bin/activate"
echo "  3. Run pipeline:    python run_pipeline.py"
echo "  4. Launch dashboard: streamlit run src/dashboard/app.py"
