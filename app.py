import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

st.set_page_config(page_title="Marketing Intelligence & Budget Optimizer", page_icon="📊", layout="wide")
CHANNELS = ["google_spend", "meta_spend", "email_spend", "influencer_spend"]
LABELS = dict(zip(CHANNELS, ["Google Ads", "Meta Ads", "Email Marketing", "Influencer Marketing"]))
OPTIONAL = ["promo", "price_index"]

def demo_data(n=156, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="W")
    t = np.arange(n)
    spends = [
        rng.uniform(18000, 65000, n)*(1+0.15*np.sin(t/10)),
        rng.uniform(15000, 55000, n)*(1+0.12*np.cos(t/13)),
        rng.uniform(4000, 18000, n),
        rng.uniform(5000, 26000, n)*(1+0.2*np.sin(t/17))
    ]
    promo = rng.binomial(1, 0.18, n)
    price = 1 + 0.04*np.sin(t/15) + rng.normal(0, 0.012, n)
    effects = []
    for spend, decay, scale, alpha in zip(spends, [0.55,0.48,0.25,0.40], [25000,22000,9000,12000], [1.15,1.1,1.0,1.2]):
        carry = np.zeros(n)
        for i in range(n):
            carry[i] = spend[i] + (decay*carry[i-1] if i else 0)
        effects.append(carry**alpha/(carry**alpha + scale**alpha))
    sales = (107000 + 55*t + 90000*effects[0] + 75000*effects[1] +
             42000*effects[2] + 55000*effects[3] + 26000*promo -
             18000*(price-1) + rng.normal(0,9500,n))
    return pd.DataFrame({"date":dates,"sales":np.maximum(sales,1000),
        **{c:s for c,s in zip(CHANNELS,spends)},"promo":promo,"price_index":price})

def clean_data(df):
    required = ["date","sales"] + CHANNELS
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError("Missing required columns: " + ", ".join(missing))
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ["sales"] + CHANNELS + OPTIONAL:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date","sales"]+CHANNELS).sort_values("date").reset_index(drop=True)
    if len(d) < 20: raise ValueError("At least 20 valid weekly observations are required.")
    if (d[CHANNELS] < 0).any().any(): raise ValueError("Spend values cannot be negative.")
    if "promo" not in d: d["promo"] = 0
    if "price_index" not in d: d["price_index"] = 1.0
    d[OPTIONAL] = d[OPTIONAL].ffill().fillna(0)
    return d

def adstock(x, decay):
    x = np.asarray(x, dtype=float); out = np.zeros(len(x))
    for i in range(len(x)): out[i] = x[i] + (decay*out[i-1] if i else 0)
    return out

def hill(x, alpha, k):
    x = np.maximum(np.asarray(x,dtype=float),0)
    xa = x**alpha
    return xa/(xa+k**alpha)

def make_features(d, decays, shapes, half):
    f = pd.DataFrame(index=d.index)
    for c in CHANNELS:
        f[c] = hill(adstock(d[c],decays[c]),shapes[c],half[c])
    f["promo"] = d["promo"].astype(float)
    f["price_index"] = d["price_index"].astype(float)
    f["trend"] = np.arange(len(d),dtype=float)
    return f

def fit(d, decays, shapes, half, ridge_alpha):
    f = make_features(d,decays,shapes,half)
    scaler = StandardScaler()
    xs = scaler.fit_transform(f)
    model = Ridge(alpha=ridge_alpha).fit(xs,d["sales"])
    return model,scaler,f,model.predict(xs)

def predict(model,scaler,f):
    return model.predict(scaler.transform(f))

def optimize(model,scaler,d,decays,shapes,half,budget,max_share):
    recent = d.tail(min(13,len(d)))
    current = recent[CHANNELS].mean().to_numpy()
    current = current/current.sum()*budget if current.sum()>0 else np.ones(4)*budget/4
    fixed_promo = float(recent["promo"].mean())
    fixed_price = float(recent["price_index"].mean())
    trend = float(len(d)-1)
    def objective(b):
        row = pd.DataFrame([{**dict(zip(CHANNELS,b)),"promo":fixed_promo,"price_index":fixed_price}])
        f = make_features(row,decays,shapes,half)
        f["trend"] = trend
        for j,c in enumerate(CHANNELS):
            f[c] = hill([b[j]/(1-decays[c])],shapes[c],half[c])[0]
        return -float(predict(model,scaler,f)[0])
    result = minimize(objective,current,method="SLSQP",
        bounds=[(0,budget*max_share)]*4,
        constraints=[{"type":"eq","fun":lambda b: np.sum(b)-budget}],
        options={"maxiter":500,"ftol":1e-7})
    if not result.success: raise RuntimeError(result.message)
    return current,result.x,-result.fun

st.title("📊 Marketing Intelligence & Budget Optimizer")
st.caption("Marketing Mix Modeling • Incrementality Analysis • Dynamic Budget Allocation")
st.info("Decision-support prototype. Model-based counterfactual estimates are not proof of causality.")

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose data",["Synthetic demo data","Upload CSV"])
    file = st.file_uploader("Upload weekly marketing CSV",type=["csv"]) if source=="Upload CSV" else None
    st.divider()
    st.header("Model assumptions")
    ridge_alpha = st.slider("Ridge regularization",0.1,100.0,10.0,0.1)
    default_decay = [0.55,0.48,0.25,0.40]
    default_half = [25000,22000,9000,12000]
    decays,shapes,half = {},{},{}
    for i,c in enumerate(CHANNELS):
        with st.expander(LABELS[c]):
            decays[c] = st.slider("Adstock decay",0.0,0.95,default_decay[i],0.01,key="d"+c)
            shapes[c] = st.slider("Hill shape (alpha)",0.5,3.0,1.0,0.1,key="a"+c)
            half[c] = st.number_input("Half-saturation scale",min_value=1.0,value=float(default_half[i]),step=1000.0,key="h"+c)

try:
    raw = pd.read_csv(file) if source=="Upload CSV" and file is not None else demo_data()
    if source=="Upload CSV" and file is None:
        st.warning("Upload a CSV, or select Synthetic demo data.")
        st.stop()
    data = clean_data(raw)
except Exception as e:
    st.error(f"Data error: {e}")
    st.markdown("Required columns: `date`, `sales`, `google_spend`, `meta_spend`, `email_spend`, `influencer_spend`. Optional: `promo`, `price_index`.")
    st.stop()

model,scaler,features,fitted = fit(data,decays,shapes,half,ridge_alpha)
r2 = r2_score(data["sales"],fitted)
rmse = np.sqrt(mean_squared_error(data["sales"],fitted))
mae = mean_absolute_error(data["sales"],fitted)
spend = data[CHANNELS].sum()
inc = {}
for c in CHANNELS:
    cf = features.copy()
    cf[c] = 0.0
    inc[c] = max(0.0,float(np.sum(fitted-predict(model,scaler,cf))))
results = pd.DataFrame({
    "Channel":[LABELS[c] for c in CHANNELS],
    "Marketing Spend":[spend[c] for c in CHANNELS],
    "Estimated Incremental Revenue":[inc[c] for c in CHANNELS],
    "Incremental ROAS":[inc[c]/spend[c] if spend[c] else 0 for c in CHANNELS]
})
total_spend = float(spend.sum()); total_sales = float(data.sales.sum())

overview,performance,incrementality,optimizer,data_tab = st.tabs(["Overview","Model Performance","Channel Incrementality","Budget Optimizer","Data"])
with overview:
    a,b,c,d = st.columns(4)
    a.metric("Total marketing spend",f"₹{total_spend:,.0f}")
    b.metric("Observed revenue",f"₹{total_sales:,.0f}")
    c.metric("Estimated incremental revenue",f"₹{sum(inc.values()):,.0f}")
    d.metric("Overall ROAS",f"{total_sales/total_spend:.2f}x" if total_spend else "—")
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=data.date,y=data.sales,name="Observed revenue"))
    chart.add_trace(go.Scatter(x=data.date,y=fitted,name="Fitted revenue"))
    chart.update_layout(title="Observed vs fitted revenue",xaxis_title="Week",yaxis_title="Revenue")
    st.plotly_chart(chart,use_container_width=True)
    st.plotly_chart(px.bar(results,x="Channel",y="Marketing Spend",title="Total spend by channel"),use_container_width=True)

with performance:
    x,y,z = st.columns(3)
    x.metric("R² (in-sample)",f"{r2:.3f}")
    y.metric("RMSE",f"₹{rmse:,.0f}")
    z.metric("MAE",f"₹{mae:,.0f}")
    st.caption("These are in-sample fit metrics, not out-of-sample forecasting scores.")
    st.plotly_chart(px.scatter(x=data.sales,y=fitted,labels={"x":"Observed revenue","y":"Fitted revenue"},title="Observed vs fitted"),use_container_width=True)
    st.plotly_chart(px.bar(x=features.columns,y=model.coef_,labels={"x":"Feature","y":"Standardized coefficient"},title="Ridge coefficients"),use_container_width=True)

with incrementality:
    st.caption("Each channel is removed separately while other inputs are held constant. Contributions can overlap, especially when channel spends are correlated; do not interpret their sum as a causal decomposition.")
    st.dataframe(results.style.format({"Marketing Spend":"₹{:,.0f}","Estimated Incremental Revenue":"₹{:,.0f}","Incremental ROAS":"{:.2f}x"}),use_container_width=True,hide_index=True)
    st.plotly_chart(px.bar(results,x="Channel",y="Incremental ROAS",title="Incremental ROAS by channel"),use_container_width=True)
    st.download_button("Download channel results",results.to_csv(index=False).encode(), "channel_effectiveness.csv","text/csv")

with optimizer:
    st.subheader("Optimize a representative weekly budget")
    suggested = max(100.0,float(data[CHANNELS].tail(13).mean().sum()))
    budget = st.number_input("Weekly budget",min_value=100.0,value=float(round(suggested,2)),step=1000.0)
    max_share = st.slider("Maximum share for any one channel",25,100,70,5)/100
    if st.button("Optimize budget allocation",type="primary"):
        try:
            cur,opt,predicted = optimize(model,scaler,data,decays,shapes,half,budget,max_share)
            allocation = pd.DataFrame({"Channel":[LABELS[c] for c in CHANNELS],
                "Current Allocation":cur,"Optimized Allocation":opt,
                "Current Share":cur/cur.sum(),"Optimized Share":opt/opt.sum()})
            st.session_state["allocation"] = allocation
            st.session_state["optimized_prediction"] = predicted
        except Exception as e:
            st.error(f"Optimization failed: {e}. Try increasing the maximum channel share.")
    if "allocation" in st.session_state:
        alloc = st.session_state["allocation"]
        l,r = st.columns(2)
        l.plotly_chart(px.pie(alloc,names="Channel",values="Current Allocation",title="Current allocation"),use_container_width=True)
        r.plotly_chart(px.pie(alloc,names="Channel",values="Optimized Allocation",title="Optimized allocation"),use_container_width=True)
        st.dataframe(alloc.style.format({"Current Allocation":"₹{:,.0f}","Optimized Allocation":"₹{:,.0f}","Current Share":"{:.1%}","Optimized Share":"{:.1%}"}),use_container_width=True,hide_index=True)
        st.metric("Optimized representative-period predicted revenue",f"₹{st.session_state['optimized_prediction']:,.0f}")
        st.download_button("Download optimized allocation",alloc.to_csv(index=False).encode(),"optimized_budget_allocation.csv","text/csv")
    else:
        st.info("Enter the weekly budget and click Optimize budget allocation.")

with data_tab:
    st.write(f"{len(data)} valid observations | {data.date.min().date()} to {data.date.max().date()}")
    st.dataframe(data,use_container_width=True,hide_index=True)
    st.download_button("Download cleaned data",data.to_csv(index=False).encode(),"cleaned_marketing_data.csv","text/csv")

with st.expander("Methodology and limitations"):
    st.markdown("""
    - Adstock uses a recursive carryover transformation; Hill saturation models diminishing returns.
    - Decay, Hill shape and half-saturation values are user-configured assumptions in this prototype.
    - Channel counterfactuals compare model predictions with one channel removed; they are not experimentally established causal effects.
    - Correlated channel spending can make individual contributions difficult to identify.
    - Model metrics are in-sample. Add time-based holdout validation before production use.
    - Budget optimization maximizes predicted revenue, not profit, and uses a representative weekly budget.
    - The optimizer does not include margin, uncertainty, or channel-specific minimum spend constraints.
    """)
