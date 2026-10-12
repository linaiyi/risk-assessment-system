import streamlit as st
import pandas as pd
import joblib

# ==================== 加载模型（使用缓存，避免重复加载） ====================
@st.cache_resource
def load_all_models():
    model_cf = joblib.load('saved_models-f/rf_best_model-f.joblib')
    scaler_cf = joblib.load('saved_models-f/scaler-f.joblib')
    features_cf = joblib.load('saved_models-f/feature_names-f.joblib')

    model_apha = joblib.load('saved_models/cat_best_model-apha.joblib')
    scaler_apha = joblib.load('saved_models/scaler-apha.joblib')
    features_apha = joblib.load('saved_models/feature_names-apha.joblib')

    model_f = joblib.load('saved_models-int1/rf_best_model-int1.joblib')
    scaler_f = joblib.load('saved_models-int1/scaler-int1.joblib')
    features_f = joblib.load('saved_models-int1/feature_names-int1.joblib')

    return model_cf, scaler_cf, features_cf, model_apha, scaler_apha, features_apha, model_f, scaler_f, features_f

# 加载模型
model_cf, scaler_cf, features_cf, model_apha, scaler_apha, features_apha, model_f, scaler_f, features_f = load_all_models()
features = features_cf

# ==================== 风险计算函数 ====================
def compute_ri(pred_cf, pred_apha, pred_f):
    # 定义各预测值的最小值和最大值（根据你提供的表）
    min_f, max_f = 0.66846, 2.039469        # 对应 pred_cf
    min_apha, max_apha = 0.220223, 6.326085 # 对应 pred_apha
    min_int1, max_int1 = 0.133038, 6.131449 # 对应 pred_f

    # Min-Max 归一化到 [0, 1]
    norm_cf = (pred_cf - min_f) / (max_f - min_f)
    norm_apha = (pred_apha - min_apha) / (max_apha - min_apha)
    norm_f = (pred_f - min_int1) / (max_int1 - min_int1)

    # 加权求和（权重不变：0.4, 0.3, 0.3）
    return 0.4 * norm_cf + 0.3 * norm_apha + 0.3 * norm_f

def get_risk_level(ri):
    if ri < 0.2:
        return "Low Risk"
    elif ri < 0.5:
        return "Medium Risk"
    elif ri < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"

# ==================== 界面构建 ====================
st.set_page_config(page_title="Multi-Target Risk Assessment System", layout="wide")

# 注入自定义 CSS（尽量保留你原来的样式）
css = """
<style>
    .stApp { background: #ffffff; }
    div[data-testid="stNumberInput"] label p { font-size: 1.1rem; font-weight: normal; }
    div[data-testid="stTextInput"] label p { font-size: 1.1rem; font-weight: normal; }
    .stButton button { width: 100%; font-size: 1.1rem; padding: 0.45rem 1.2rem; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.title("🔬 Comprehensive Prediction")
st.write("Please enter the following feature values (all units: mg/L):")

# 使用表单（Form）来避免每次输入都触发整个页面重跑
with st.form("prediction_form"):
    # 分成两列输入
    col1, col2 = st.columns(2)
    
    input_values = []
    half = len(features) // 2

    with col1:
        for name in features[:half]:
            val = st.number_input(label=name, value=0.0, format="%.4f", key=f"input_{name}")
            input_values.append(val)

    with col2:
        for name in features[half:]:
            val = st.number_input(label=name, value=0.0, format="%.4f", key=f"input_{name}")
            input_values.append(val)

    # 提交按钮
    submitted = st.form_submit_button("Predict", type="primary")

# ==================== 预测与输出 ====================
if submitted:
    # 构建 DataFrame
    input_dict = {col: [val] for col, val in zip(features, input_values)}
    df = pd.DataFrame(input_dict)
    
    # 预测
    pred_cf = model_cf.predict(scaler_cf.transform(df))[0]
    pred_apha = model_apha.predict(scaler_apha.transform(df))[0]
    pred_f = model_f.predict(scaler_f.transform(df))[0]
    
    # 计算风险指数
    ri = compute_ri(pred_cf, pred_apha, pred_f)
    risk_level = get_risk_level(ri)
    
    # 显示结果
    st.subheader("Prediction Results")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.text_input(label="Conjugation Frequency (RF)", value=f"{pred_cf:.4f}", disabled=True)
        st.text_input(label="aphA (AdaBoost)", value=f"{pred_apha:.4f}", disabled=True)
        st.text_input(label="int1 (RF)", value=f"{pred_f:.4f}", disabled=True)
    with res_col2:
        st.text_input(label="Risk Index (RI)", value=f"{ri:.4f}", disabled=True)
        st.text_input(label="Risk Level", value=f"{risk_level}", disabled=True)
