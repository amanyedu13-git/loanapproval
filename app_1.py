import streamlit as st
import numpy as np
import joblib
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main {
    background-color: #0f1117;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
    border: 1px solid #2a2f3e;
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}

.hero h1 {
    font-size: 2.8rem;
    color: #f0c040;
    margin-bottom: 0.5rem;
}

.hero p {
    color: #8892a4;
    font-size: 1.1rem;
    font-weight: 300;
}

.card {
    background: #1a1f2e;
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}

.card h3 {
    color: #f0c040;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

.result-approved {
    background: linear-gradient(135deg, #0d2b1f, #1a3a2a);
    border: 2px solid #2ecc71;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-rejected {
    background: linear-gradient(135deg, #2b0d0d, #3a1a1a);
    border: 2px solid #e74c3c;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-approved h2 {
    color: #2ecc71;
    font-size: 2rem;
}

.result-rejected h2 {
    color: #e74c3c;
    font-size: 2rem;
}

.result-approved p, .result-rejected p {
    color: #aab;
    font-size: 1rem;
    margin-top: 0.5rem;
}

.metric-box {
    background: #0f1117;
    border: 1px solid #2a2f3e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-box .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f0c040;
}

.metric-box .label {
    font-size: 0.8rem;
    color: #8892a4;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #c8d0de !important;
    font-weight: 400;
}

.stButton > button {
    background: linear-gradient(135deg, #f0c040, #e0a820);
    color: #0f1117;
    font-weight: 700;
    font-size: 1.1rem;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: all 0.3s ease;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.03em;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(240, 192, 64, 0.3);
}

.tip-box {
    background: #1a1f2e;
    border-left: 3px solid #f0c040;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #8892a4;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists('best_model.pkl'):
        model = joblib.load('best_model.pkl')
        return model
    return None

model = load_model()

# ── Hero Section ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏦 Loan Approval</h1>
    <p>AI-powered prediction system — know your chances instantly</p>
</div>
""", unsafe_allow_html=True)

# ── Model Status ──────────────────────────────────────────────
if model is None:
    st.error("⚠️ Model not found! Please run the notebook first to generate `best_model.pkl`")
    st.stop()

# ── Input Form ────────────────────────────────────────────────
st.markdown('<div class="card"><h3>👤 Personal Details</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    education = st.selectbox(
        "Education Level",
        options=["Graduate", "Not Graduate"]
    )
with col2:
    dependents = st.selectbox(
        "Number of Dependents",
        options=[0, 1, 2, 3, 4, 5]
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><h3>💰 Financial Details</h3>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    income = st.number_input(
        "Annual Income (₹)",
        min_value=100000,
        max_value=10000000,
        value=5000000,
        step=100000,
        help="Your yearly income in Rupees"
    )
with col4:
    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=100000,
        max_value=50000000,
        value=15000000,
        step=100000,
        help="Total loan amount you need"
    )

cibil = st.slider(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700,
    step=1,
    help="Credit score between 300-900. Higher is better!"
)

st.markdown('</div>', unsafe_allow_html=True)

# ── Live Metrics ──────────────────────────────────────────────
ratio = loan_amount / income
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="value">{cibil}</div>
        <div class="label">CIBIL Score</div>
    </div>""", unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="value">{ratio:.1f}x</div>
        <div class="label">Loan/Income Ratio</div>
    </div>""", unsafe_allow_html=True)

with col_m3:
    cibil_status = "Excellent 🟢" if cibil >= 750 else ("Good 🟡" if cibil >= 650 else "Poor 🔴")
    st.markdown(f"""
    <div class="metric-box">
        <div class="value" style="font-size:1.1rem">{cibil_status}</div>
        <div class="label">Credit Status</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────
if st.button("🔍 Predict Loan Approval"):

    # Prepare input
    edu_encoded = 1 if education == "Graduate" else 0
    loan_income_ratio = loan_amount / income

    user_input = np.array([[dependents, edu_encoded, income,
                            loan_amount, cibil, loan_income_ratio]])

    prediction = model.predict(user_input)[0]
    proba = model.predict_proba(user_input)[0]
    confidence = max(proba) * 100

    if prediction == 1:
        st.markdown(f"""
        <div class="result-approved">
            <h2>✅ Loan Approved!</h2>
            <p>Confidence: <strong>{confidence:.1f}%</strong></p>
            <p>Congratulations! Based on your profile, your loan is likely to be approved.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
            <h2>❌ Loan Rejected</h2>
            <p>Confidence: <strong>{confidence:.1f}%</strong></p>
            <p>Based on your profile, your loan may not be approved. Try improving your CIBIL score.</p>
        </div>
        """, unsafe_allow_html=True)

    # Tips
    st.markdown("<br>", unsafe_allow_html=True)
    if cibil < 700:
        st.markdown("""
        <div class="tip-box">
            💡 <strong>Tip:</strong> Your CIBIL score is low. Pay existing EMIs on time to improve it above 750.
        </div>
        """, unsafe_allow_html=True)
    if ratio > 5:
        st.markdown("""
        <div class="tip-box">
            💡 <strong>Tip:</strong> Your loan amount is very high compared to income. Consider reducing the loan amount.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#4a5060; font-size:0.85rem;">
    Built with ❤️ using Random Forest & Streamlit
</div>
""", unsafe_allow_html=True)
