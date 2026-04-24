# uhpc-streamlit-deploy

Streamlit deployment package for the **UHPC Inverse Design** application.

## Files

| File / Folder | Purpose |
|---|---|
| `streamlit_app.py` | Main Streamlit dashboard — UI, charts, and session state |
| `inverse_design.py` | Core engine — BNN model definition, ensemble loading, SLSQP optimizer |
| `requirements.txt` | Python package dependencies |
| `.streamlit/config.toml` | Theme config (light mode, navy accent palette) |
| `Data/` | Training data (features + targets) and material property database |
| `models/checkpoints/` | 5-fold BNN checkpoints (`.ckpt`) |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

App runs at `http://localhost:8501`.

## Deploy to Streamlit Cloud

1. Fork / push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path** to `uhpc-streamlit-deploy/streamlit_app.py`
4. Click **Deploy**

> See the [root README](../README.md) for full project documentation.
