# UNLOX Project - How to Run (Step by Step)

This file has instructions for both Mac and Windows. Jump to whichever section matches your computer.

---

## What this project is

This is a YouTube channel analytics project. It pulls video data from YouTube (or uses fake data if no API key), computes viral scores, runs sentiment analysis on comments, trains a machine learning model, and shows everything in a Streamlit dashboard with charts.

---

---

# MAC INSTRUCTIONS

---

## What you need (Mac)

- Python 3.11 installed
- The project folder somewhere on your computer

To check if Python 3.11 is installed, open Terminal and run:
```
python3.11 --version
```
It should print: Python 3.11.x

If it says command not found, download it from https://www.python.org/downloads/ — look for the 3.11 version specifically.

---

## STEP 1 — Open Terminal and go to the project folder

```
cd /path/to/unlox
```

For example if the folder is on your Desktop:
```
cd ~/Desktop/unlox
```

All commands from this point must be run from inside this folder.

---

## STEP 2 — Create the virtual environment (only once)

```
python3.11 -m venv venv
```

This creates a folder called `venv` inside the project. Takes about 15 seconds.

---

## STEP 3 — Install all packages (only once)

```
venv/bin/pip install -r requirements.txt
```

Takes 3-5 minutes. When done you will see: `Successfully installed ...`

If you see red errors try:
```
venv/bin/pip install -r requirements.txt --prefer-binary
```

---

## STEP 4 — Set up your API key (optional)

If you skip this the project uses dummy data and still works fine.

Copy the example file:
```
cp .env.example .env
```

Open `.env` in any text editor and fill in:
```
YOUTUBE_API_KEY=your_actual_api_key_here
YOUTUBE_CHANNEL_ID=your_channel_id_here
```

How to get an API key:
1. Go to https://console.cloud.google.com
2. Create a new project
3. Go to APIs & Services → Library → search "YouTube Data API v3" → Enable it
4. Go to APIs & Services → Credentials → Create Credentials → API Key
5. Copy the key and paste it into .env

How to find your channel ID:
1. Go to your YouTube channel in the browser
2. The URL looks like: youtube.com/channel/UCxxxxxxxx — that part after /channel/ is your ID
3. If your URL uses @username instead, go to your channel → Settings → Advanced settings → Channel ID

---

## STEP 5 — Run the pipeline

```
venv/bin/python run_pipeline.py
```

Takes 1-3 minutes. You will see each step printing in the terminal. When done it says "Pipeline done."

---

## STEP 6 — Open the dashboard

```
venv/bin/streamlit run src/dashboard/app.py
```

Browser opens automatically at http://localhost:8501

If it doesn't open automatically, go to that URL manually in your browser.

Use the left sidebar to switch between the 9 pages.

To stop: press `Ctrl + C` in Terminal.

---

## Mac — Run again next time

```
cd /path/to/unlox
venv/bin/streamlit run src/dashboard/app.py
```

With fresh data:
```
cd /path/to/unlox
venv/bin/python run_pipeline.py
venv/bin/streamlit run src/dashboard/app.py
```

---

## Mac — Run one stage at a time

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

## Mac — Common errors

**command not found: python3.11**
Download Python 3.11 from https://www.python.org/downloads/

**Port 8501 is already in use**
```
lsof -i :8501
kill <PID number shown>
```

**No module named 'dotenv'**
```
rm -rf venv
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt --prefer-binary
```

**videos.csv not found**
Run the pipeline first: `venv/bin/python run_pipeline.py`

**scipy failed to build**
Make sure you are using Python 3.11 and not 3.12/3.13/3.14. Check with `python3.11 --version`

---

## Mac — Full commands from scratch

```
cd /path/to/unlox
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python run_pipeline.py
venv/bin/streamlit run src/dashboard/app.py
```

---

---

# WINDOWS INSTRUCTIONS

---

## What you need (Windows)

- Windows 10 or Windows 11
- Python 3.11 installed
- The project folder somewhere on your computer
- Command Prompt or PowerShell (both work)

To check if Python 3.11 is installed, open Command Prompt and run:
```
python --version
```
It should print: Python 3.11.x

If it says "not recognized" or shows a different version, install Python 3.11:

1. Go to https://www.python.org/downloads/
2. Click "Download Python 3.11.x" (look for 3.11 specifically, not 3.12 or higher)
3. Run the installer
4. **Important:** On the first screen of the installer, check the box that says "Add Python to PATH" before clicking Install Now. If you miss this the commands will not work.
5. After install, close all Command Prompt windows and open a new one
6. Run `python --version` again to confirm

---

## STEP 1 — Open Command Prompt and go to the project folder

Press `Windows + R`, type `cmd`, press Enter. This opens Command Prompt.

Then navigate to the project folder:
```
cd C:\path\to\unlox
```

For example if the folder is on your Desktop:
```
cd C:\Users\YourName\Desktop\unlox
```

Replace `YourName` with your actual Windows username.

To check you are in the right folder:
```
dir
```

You should see files like `run_pipeline.py`, `requirements.txt`, `config` folder etc.

---

## STEP 2 — Create the virtual environment (only once)

```
python -m venv venv
```

This creates a `venv` folder inside the project. Takes about 15 seconds.

---

## STEP 3 — Activate the virtual environment

On Windows you must activate the venv before using it. Run:

```
venv\Scripts\activate
```

After running this you will see `(venv)` appear at the start of your command prompt line like this:
```
(venv) C:\Users\YourName\Desktop\unlox>
```

This means the venv is active. You need to do this every time you open a new Command Prompt window to work on this project.

---

## STEP 4 — Install all packages (only once)

Make sure the venv is activated (you see `(venv)` in the prompt), then run:

```
pip install -r requirements.txt
```

Takes 3-5 minutes. When done you will see: `Successfully installed ...`

If you see red errors try:
```
pip install -r requirements.txt --prefer-binary
```

---

## STEP 5 — Set up your API key (optional)

If you skip this the project uses dummy data and still works fine.

Copy the example env file:
```
copy .env.example .env
```

Open the `.env` file in Notepad:
```
notepad .env
```

Fill in your values:
```
YOUTUBE_API_KEY=your_actual_api_key_here
YOUTUBE_CHANNEL_ID=your_channel_id_here
```

Save and close Notepad.

How to get an API key:
1. Go to https://console.cloud.google.com
2. Create a new project
3. Go to APIs & Services → Library → search "YouTube Data API v3" → Enable it
4. Go to APIs & Services → Credentials → Create Credentials → API Key
5. Copy the key and paste it into .env

How to find your channel ID:
1. Go to your YouTube channel in the browser
2. The URL looks like: youtube.com/channel/UCxxxxxxxx — that part after /channel/ is your ID
3. If your URL uses @username instead, go to your channel → Settings → Advanced settings → Channel ID

---

## STEP 6 — Run the pipeline

Make sure `(venv)` is showing in your prompt, then run:

```
python run_pipeline.py
```

Takes 1-3 minutes. You will see each step printing in the terminal. When done it says "Pipeline done."

---

## STEP 7 — Open the dashboard

```
streamlit run src/dashboard/app.py
```

Browser opens automatically at http://localhost:8501

If it doesn't open automatically, open Chrome or Edge and go to:
```
http://localhost:8501
```

Use the left sidebar to switch between the 9 pages:
1. Overview
2. Engagement Trends
3. Viral Score
4. Sentiment Analysis
5. Best Posting Time
6. A/B Testing
7. Virality Predictor
8. Recommendations
9. Trend Forecast

To stop the dashboard: press `Ctrl + C` in Command Prompt.

---

## Windows — Run again next time

Every time you open a new Command Prompt you must activate the venv first:

```
cd C:\path\to\unlox
venv\Scripts\activate
streamlit run src/dashboard/app.py
```

With fresh data:
```
cd C:\path\to\unlox
venv\Scripts\activate
python run_pipeline.py
streamlit run src/dashboard/app.py
```

---

## Windows — Run one stage at a time

Make sure venv is activated first, then:

```
python run_pipeline.py --stage extract
python run_pipeline.py --stage score
python run_pipeline.py --stage sentiment
python run_pipeline.py --stage ab_test
python run_pipeline.py --stage virality
python run_pipeline.py --stage recommend
python run_pipeline.py --stage forecast
```

Run them in this order because each stage needs the output of the previous one.

---

## Windows — Common errors

**'python' is not recognized as an internal or external command**

Python is not installed or was not added to PATH. Reinstall Python 3.11 from https://www.python.org/downloads/ and make sure to check "Add Python to PATH" on the first screen of the installer.

**'pip' is not recognized**

The venv is not activated. Run `venv\Scripts\activate` first.

**venv\Scripts\activate gives an error about execution policy**

Windows PowerShell blocks scripts by default. Either switch to Command Prompt (cmd) instead of PowerShell, or run this in PowerShell to allow it:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating again.

**Port 8501 is already in use**

Open Task Manager (Ctrl + Shift + Esc), go to Details tab, find `python.exe`, right-click and End Task. Then run streamlit again.

Or in Command Prompt:
```
netstat -ano | findstr :8501
taskkill /PID <number shown> /F
```

**No module named 'dotenv'**

The venv is not activated or packages are not installed. Make sure you see `(venv)` in the prompt, then:
```
pip install -r requirements.txt --prefer-binary
```

**ERROR: Could not build wheels for scipy**

You are using Python 3.12 or higher. Only Python 3.11 works. Uninstall your current Python, install 3.11 from https://www.python.org/downloads/, then delete the venv folder and start from Step 2.

**videos.csv not found**

Run the pipeline first: `python run_pipeline.py`

**Prophet install fails on Windows**

Prophet can be tricky on Windows. Try:
```
pip install pystan==2.19.1.1
pip install prophet --prefer-binary
```
If it still fails, the project will automatically use ETS forecasting instead. Everything still works.

---

## Windows — Full commands from scratch

Open Command Prompt and run these one by one:

```
cd C:\path\to\unlox
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run src/dashboard/app.py
```

---

## Windows — Quick reference

| What you want to do | Command |
|---------------------|---------|
| Go to project folder | `cd C:\path\to\unlox` |
| Activate venv | `venv\Scripts\activate` |
| Create venv (first time only) | `python -m venv venv` |
| Install packages (first time only) | `pip install -r requirements.txt` |
| Run full pipeline | `python run_pipeline.py` |
| Run one stage | `python run_pipeline.py --stage extract` |
| Open dashboard | `streamlit run src/dashboard/app.py` |
| Stop dashboard | `Ctrl + C` |

---

## Key difference between Mac and Windows commands

| Action | Mac | Windows |
|--------|-----|---------|
| Navigate to folder | `cd ~/Desktop/unlox` | `cd C:\Users\Name\Desktop\unlox` |
| Create venv | `python3.11 -m venv venv` | `python -m venv venv` |
| Activate venv | not needed every time | `venv\Scripts\activate` (every new window) |
| Run pip | `venv/bin/pip install ...` | `pip install ...` (after activating) |
| Run python | `venv/bin/python run_pipeline.py` | `python run_pipeline.py` (after activating) |
| Run streamlit | `venv/bin/streamlit run ...` | `streamlit run ...` (after activating) |
| Copy a file | `cp .env.example .env` | `copy .env.example .env` |
| Delete a folder | `rm -rf venv` | `rmdir /s /q venv` |
| Open a file in editor | `open -e .env` | `notepad .env` |
