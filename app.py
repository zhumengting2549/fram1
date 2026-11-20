import streamlit as st
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ====================== 字体兜底配置（不再读取本地ttf文件）======================
plt.rcParams['axes.unicode_minus'] = False
# 尝试云环境可用中文字体，按顺序匹配
try:
    plt.rcParams['font.sans-serif'] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    font_ok = True
except:
    font_ok = False

# ====================== 页面基础设置 ======================
st.set_page_config(page_title="水稻玉米种植决策辅助智能体", layout="wide")
st.title("水稻玉米种植决策辅助智能体")
st.markdown("基于可解释AI，辅助小农户做产量、收益种植决策")

# ====================== 模拟样本数据 ======================
np.random.seed(42)
n = 200
data = pd.DataFrame({
    "Fertilizer": np.random.uniform(20, 80, n),
    "Irrigation": np.random.uniform(300, 800, n),
    "Temperature": np.random.uniform(18, 32, n),
    "Rainfall": np.random.uniform(50, 250, n),
    "Soil": np.random.uniform(4, 9, n),
})
data["Profit"] = (
    data["Fertilizer"] * 2.2
    + data["Irrigation"] * 0.12
    + data["Temperature"] * 8
    + data["Soil"] * 25
    - data["Rainfall"] * 0.3
    + np.random.normal(0, 12, n)
)
X = data.drop("Profit", axis=1)
y = data["Profit"]

# 训练模型
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ====================== 侧边栏输入模块 ======================
st.sidebar.header("📝 种植参数输入")
fer = st.sidebar.slider("施肥量", 20.0, 80.0, 45.0)
water = st.sidebar.slider("灌溉水量", 300.0, 800.0, 500.0)
temp = st.sidebar.slider("气温均值", 18.0, 32.0, 24.0)
rain = st.sidebar.slider("降水", 50.0, 250.0, 120.0)
soil = st.sidebar.slider("土壤肥力", 4.0, 9.0, 6.5)

input_df = pd.DataFrame([[fer, water, temp, rain, soil]], columns=X.columns)
pred = model.predict(input_df)[0]
st.subheader("📊 预测结果")
st.metric("预估收益(元/亩)", f"{pred:.2f}")

# ====================== SHAP因子重要性柱状图 ======================
st.subheader("图2：SHAP因子影响分析（可解释AI核心）")
shap_importance = np.abs(shap_values).mean(axis=0)
feature_names = ["Fertilizer","Irrigation","Temperature","Rainfall","Soil"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(feature_names, shap_importance)
ax.set_title("SHAP Feature Importance")
ax.set_xlabel("SHAP Value")
ax.invert_yaxis()
st.pyplot(fig)

st.markdown("---")
st.caption("第二届全国人工智能应用创新大赛｜小农户产量收益双目标可解释助农Agent")