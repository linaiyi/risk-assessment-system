import numpy as np
import pandas as pd
import joblib
import gradio as gr

# ==================== 加载模型 ====================
model_cf = joblib.load('saved_models-f/rf_best_model-f.joblib')
scaler_cf = joblib.load('saved_models-f/scaler-f.joblib')
features_cf = joblib.load('saved_models-f/feature_names-f.joblib')

model_apha = joblib.load('saved_models/cat_best_model-apha.joblib')
scaler_apha = joblib.load('saved_models/scaler-apha.joblib')
features_apha = joblib.load('saved_models/feature_names-apha.joblib')

model_f = joblib.load('saved_models-int1/rf_best_model-int1.joblib')
scaler_f = joblib.load('saved_models-int1/scaler-int1.joblib')
features_f = joblib.load('saved_models-int1/feature_names-int1.joblib')

features = features_cf

# ==================== 风险计算函数 ====================
def compute_ri(pred_cf, pred_apha, pred_f):
    return 0.4 * pred_cf + 0.3 * pred_apha + 0.3 * pred_f

def get_risk_level(ri):
    if ri < 0.2:
        return "Low Risk"
    elif ri < 0.5:
        return "Medium Risk"
    elif ri < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"

# ==================== 预测函数 ====================
def predict_all(*args):
    input_dict = {col: [val] for col, val in zip(features, args)}
    df = pd.DataFrame(input_dict)
    pred_cf = model_cf.predict(scaler_cf.transform(df))[0]
    pred_apha = model_apha.predict(scaler_apha.transform(df))[0]
    pred_f = model_f.predict(scaler_f.transform(df))[0]
    ri = compute_ri(pred_cf, pred_apha, pred_f)
    risk_level = get_risk_level(ri)
    return (
        f"{pred_cf:.4f}",
        f"{pred_apha:.4f}",
        f"{pred_f:.4f}",
        f"{ri:.4f}",
        f"{risk_level}"
    )

# ==================== CSS ====================
css = """
* {
    font-family: Arial, Helvetica, sans-serif !important;
    box-sizing: border-box !important;
    color: #000000 !important;
}

body {
    background: #ffffff;
}

.gr-column, .gr-box, .gr-form, .gr-panel, .gr-group, .gr-block, .contain, .wrap, .gradio-container {
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* 最外层大容器：整体左对齐，宽度由内容决定 */
.wrapper {
    width: fit-content !important;
    max-width: 100% !important;
    margin: 0 auto 0 0 !important;
    padding: 0 !important;
}

/* 主横向行：输入两列 + 输出列 */
.main-row {
    width: fit-content !important;
    max-width: 100% !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    gap: 1.5rem !important;
    flex-wrap: nowrap !important;
}

/* 每一列固定宽度，防止重叠 */
.input-col,
.output-col {
    width: 150px !important;
    max-width: 150px !important;
    min-width: 150px !important;
    gap: 0.35rem !important;
}

/* 输出下方的按钮容器，和输出列同宽 */
.output-btn-col {
    width: 150px !important;
    max-width:150px !important;
    min-width:150px !important;
    gap:1rem !important;
    margin-top:1rem !important;
}

/* 统一表单项 */
.gr-form-item {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
}

/* label放大并防止折行太严重 */
.gr-form-item label {
    font-weight: normal !important;
    font-size: 1.3rem !important;
    margin: 0 0 0.2rem 0 !important;
    padding: 0 !important;
    line-height: 1.25 !important;
    white-space: normal !important;
    word-break: break-word !important;
}

/* 输入框、输出框 */
.gr-form-item input,
.gr-form-item textarea {
    width: 100% !important;
    font-size: 1.1rem !important;
    padding: 0.3rem 0.45rem !important;
    border: 1px solid #cccccc !important;
    border-radius: 0.25rem !important;
    background: #fff !important;
    min-height: 2.2rem !important;
    margin: 0 !important;
}

/* 按钮样式微调 */
.gr-button {
    font-size: 1.1rem !important;
    padding: 0.45rem 1.2rem !important;
    border-radius: 0.35rem !important;
    width:100% !important;
}
"""

# ==================== 界面构建 ====================
with gr.Blocks(
    title="Multi‑Target Risk Assessment System",
    css=css,
    theme=gr.themes.Base()
) as demo:
    gr.Markdown("# 🔬 Comprehensive Prediction")
    gr.Markdown("Please enter the following feature values (all units: mg/L):")

    with gr.Column(elem_classes=["wrapper"], scale=0):
        with gr.Row(elem_classes=["main-row"], equal_height=False):
            # 输入左列
            with gr.Column(elem_classes=["input-col"], scale=0):
                inputs = []
                half = len(features) // 2
                for name in features[:half]:
                    n = gr.Number(label=name, value=0.0, precision=4)
                    inputs.append(n)

            # 输入右列
            with gr.Column(elem_classes=["input-col"], scale=0):
                for name in features[half:]:
                    n = gr.Number(label=name, value=0.0, precision=4)
                    inputs.append(n)

            # 输出列 + Predict按钮（仅保留Predict，放在输出框正下方）
            with gr.Column(scale=0):
                with gr.Column(elem_classes=["output-col"]):
                    out_cf = gr.Textbox(label="Conjugation Frequency (RF)", lines=1)
                    out_apha = gr.Textbox(label="aphA (AdaBoost)", lines=1)
                    out_int1 = gr.Textbox(label="int1 (RF)", lines=1)
                    out_ri = gr.Textbox(label="Risk Index (RI)", lines=1)
                    out_risk = gr.Textbox(label="Risk Level", lines=1)
                # 仅Predict按钮
                with gr.Column(elem_classes=["output-btn-col"]):
                    predict_btn = gr.Button("Predict", variant="primary")

    predict_btn.click(
        predict_all,
        inputs=inputs,
        outputs=[out_cf, out_apha, out_int1, out_ri, out_risk]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
