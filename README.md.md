# Multi-Target Risk Assessment System

基于 Streamlit 的多目标风险评估在线预测系统。

## 项目简介
输入特征值后，系统会自动预测以下指标并计算综合风险等级：
- Conjugation Frequency (RF)
- aphA (AdaBoost)
- int1 (RF)
- Risk Index (RI)
- Risk Level

## 技术栈
- Python
- Streamlit
- scikit-learn
- CatBoost
- joblib
- pandas

## 项目结构
```text
.
├── app.py                          # Streamlit 主程序
├── requirements.txt                # Python 依赖清单
├── README.md                       # 项目说明文档
├── saved_models-f/                 # RF 模型文件夹
│   ├── rf_best_model-f.joblib
│   ├── scaler-f.joblib
│   └── feature_names-f.joblib
├── saved_models/                   # aphA 模型文件夹
│   ├── cat_best_model-apha.joblib
│   ├── scaler-apha.joblib
│   └── feature_names-apha.joblib
└── saved_models-int1/              # int1 模型文件夹
    ├── rf_best_model-int1.joblib
    ├── scaler-int1.joblib
    └── feature_names-int1.joblib
