# Marketing Intelligence & Budget Optimizer

Streamlit prototype for marketing mix modelling, channel incrementality analysis and constrained budget allocation.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV format
Required columns: `date`, `sales`, `google_spend`, `meta_spend`, `email_spend`, `influencer_spend`.
Optional columns: `promo`, `price_index`.
Each row should represent one weekly observation. At least 20 valid rows are required. Monetary values are interpreted in the units supplied; synthetic demo data use INR.

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `README.md` to GitHub.
2. Sign in to Streamlit Community Cloud and create an app from the repository.
3. Select branch `main`, set main file path to `app.py`, and deploy.

## Limitations
Adstock and Hill parameters are user-configured assumptions, not learned channel-specific effects. Counterfactual estimates are model-based, not experimentally proven causal effects. Fit metrics are in-sample. Optimization maximizes predicted revenue, not profit. Validate with time-based holdouts and experiments before operational use.
