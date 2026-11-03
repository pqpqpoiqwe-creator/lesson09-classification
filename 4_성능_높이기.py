import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실", page_icon="⚖️", layout="wide"
)

# 데이터 로드 함수 (동일 주소, UTF-8)
DATA_URL = "https://raw.githubusercontent.com/pqpqpoiqwe-creator/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
  df = pd.read_csv(DATA_URL, encoding="utf-8")
  return df


try:
  df = load_data()
except Exception as e:
  st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
  st.stop()

st.title("⚖️ 클래스 불균형 가중치 조정 및 성능 높이기 실습실")
st.markdown("---")

# 1. 속성 선택 설정 (우리말 표시, 기본값 4개)
feature_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

selected_kor_features = st.multiselect(
    "모델 학습에 사용할 속성을 고르세요 (2개 이상 선택):",
    options=list(feature_map.values()),
    default=["나이", "평균 혈당", "고혈압", "심장병"],
)

if len(selected_kor_features) < 2:
  st.warning("⚠️ 속성을 **2개 이상** 선택해 주세요.")
  st.stop()

kor_to_eng = {v: k for k, v in feature_map.items()}
selected_features = [kor_to_eng[k] for k in selected_kor_features]

# 2. 전처리 및 데이터 분할
work_df = df.copy()
if "id" in work_df.columns:
  work_df = work_df.sort_values(by="id").reset_index(drop=True)
else:
  work_df = work_df.reset_index(drop=True)
  work_df.rename(columns={"index": "id"}, inplace=True)

test_indices = []
train_indices = []

for i in range(0, len(work_df), 10):
  group = work_df.iloc[i : i + 10]
  if len(group) >= 3:
    test_indices.extend(group.index[:3].tolist())
    train_indices.extend(group.index[3:].tolist())
  else:
    test_indices.extend(group.index.tolist())

train_df = work_df.loc[train_indices].copy()
test_df = work_df.loc[test_indices].copy()

# bmi 결측치 처리 (훈련용 중앙값)
if "bmi" in work_df.columns:
  train_bmi_median = train_df["bmi"].median()
  train_df["bmi"] = train_df["bmi"].fillna(train_bmi_median)
  test_df["bmi"] = test_df["bmi"].fillna(train_bmi_median)

X_train = train_df[selected_features]
y_train = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

# 3. 모델 학습 (가중치 전 vs 후)
# 가중치 주기 전 (Default)
lr_before = LogisticRegression(random_state=42, class_weight=None)
dt_before = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42, class_weight=None
)

# 가중치 준 후 (Balanced)
lr_after = LogisticRegression(random_state=42, class_weight="balanced")
dt_after = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42, class_weight="balanced"
)

lr_before.fit(X_train, y_train)
dt_before.fit(X_train, y_train)
lr_after.fit(X_train, y_train)
dt_after.fit(X_train, y_train)

# 예측
pred_lr_b = lr_before.predict(X_test)
pred_lr_a = lr_after.predict(X_test)
pred_dt_b = dt_before.predict(X_test)
pred_dt_a = dt_after.predict(X_test)


# 평가지표 및 혼동행렬 계산 함수
def get_metrics(y_true, y_pred):
  tp = int(((y_true == 1) & (y_pred == 1)).sum())
  fp = int(((y_true == 0) & (y_pred == 1)).sum())
  fn = int(((y_true == 1) & (y_pred == 0)).sum())
  tn = int(((y_true == 0) & (y_pred == 0)).sum())
  total = len(y_true)

  acc = (tp + tn) / total if total > 0 else 0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0
  pred_pos = tp + fp
  if pred_pos == 0:
    prec = None
    f1 = None
  else:
    prec = tp / pred_pos
    f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
  return tp, fp, fn, tn, acc, recall, prec, f1


lr_b_metrics = get_metrics(y_test, pred_lr_b)
lr_a_metrics = get_metrics(y_test, pred_lr_a)
dt_b_metrics = get_metrics(y_test, pred_dt_b)
dt_a_metrics = get_metrics(y_test, pred_dt_a)

st.subheader("1. 가중치 적용 전후 모델 성능 및 혼동행렬 비교")
st.markdown(
    "뇌졸중 환자 데이터가 매우 적어 발생하는 불균형 문제를 가중치 조정(`balanced`)"
    "을 통해 해결한 결과를 비교합니다."
)


def render_comparison_section(
    model_title, metrics_b, metrics_a, pred_b, pred_a
):
  st.markdown(f"### 🤖 {model_title}")
  col1, col2 = st.columns(2)

  tp_b, fp_b, fn_b, tn_b, acc_b, rec_b, prec_b, f1_b = metrics_b
  tp_a, fp_a, fn_a, tn_a, acc_a, rec_a, prec_a, f1_a = metrics_a

  with col1:
    st.markdown("##### 📌 가중치 적용 전 (Before)")
    cm_b_df = pd.DataFrame(
        [
            [f"TN: {tn_b}", f"FP: {fp_b}"],
            [f"FN: {fn_b}", f"TP: {tp_b}"],
        ],
        index=["실제: 아님(0)", "실제: 뇌졸중(1)"],
        columns=["예측: 아님(0)", "예측: 뇌졸중(1)"],
    )
    st.dataframe(cm_b_df, use_container_width=True)

    def fmt(v, is_pct=True):
      if v is None:
        return "계산할 수 없음"
      return f"{v * 100:.2f}%" if is_pct else f"{v:.4f}"

    st.markdown(
        f"- 정확도: **{fmt(acc_b)}**<br>- 재현율: **{fmt(rec_b)}**<br>- 정밀도:"
        f" **{fmt(prec_b)}**<br>- F1 스코어: **{fmt(f1_b, False)}**",
        unsafe_allow_html=True,
    )

  with col2:
    st.markdown("##### 📌 가중치 적용 후 (After - Balanced)")
    cm_a_df = pd.DataFrame(
        [
            [f"TN: {tn_a}", f"FP: {fp_a}"],
            [f"FN: {fn_a}", f"TP: {tp_a}"],
        ],
        index=["실제: 아님(0)", "실제: 뇌졸중(1)"],
        columns=["예측: 아님(0)", "예측: 뇌졸중(1)"],
    )
    st.dataframe(cm_a_df, use_container_width=True)

    st.markdown(
        f"- 정확도: **{fmt(acc_a)}**<br>- 재현율: **{fmt(rec_a)}**<br>- 정밀도:"
        f" **{fmt(prec_a)}**<br>- F1 스코어: **{fmt(f1_a, False)}**",
        unsafe_allow_html=True,
    )

  # 환자 수 변화 요약 한 줄
  st.info(
      f"📊 **결과 요약**: 찾은 환자(TP)는 **{tp_b}명 ➔ {tp_a}명**, 놓친"
      f" 환자(FN)는 **{fn_b}명 ➔ {fn_a}명**, 헛짚은 사람(FP)은 **{fp_b}명 ➔"
      f" {fp_a}명**으로 변경되었습니다."
  )
  agent_separator = "---"
  st.markdown(agent_separator)


render_comparison_section(
    "로지스틱 회귀 (확률로 답하는 모델)",
    lr_b_metrics,
    lr_a_metrics,
    pred_lr_b,
    pred_lr_a,
)
render_comparison_section(
    "의사결정트리 (질문으로 답하는 모델)",
    dt_b_metrics,
    dt_a_metrics,
    pred_dt_b,
    pred_dt_a,
)

# 4. 시각화 (산점도 및 경계선/평면 겹쳐 그리기)
st.subheader("2. 가중치 적용 전후 결정 경계(평면) 시각화")

if len(selected_kor_features) == 2:
  plot_kor_x = selected_kor_features[0]
  plot_kor_y = selected_kor_features[1]
  x_feat = kor_to_eng[plot_kor_x]
  y_feat = kor_to_eng[plot_kor_y]

  test_df["stroke_label"] = test_df["stroke"].map(
      {0: "뇌졸중 아님 (0)", 1: "뇌졸중 (1)"}
  )

  fig = px.scatter(
      test_df,
      x=x_feat,
      y=y_feat,
      color="stroke_label",
      color_discrete_map={"뇌졸중 아님 (0)": "#1f77b4", "뇌졸중 (1)": "#ff7f0e"},
      labels={x_feat: plot_kor_x, y_feat: plot_kor_y},
      title=(
          f"로지스틱 회귀 가중치 전후 0.5 경계선 비교 ({plot_kor_x} vs"
          f" {plot_kor_y})"
      ),
      opacity=0.8,
  )

  x_min, x_max = test_df[x_feat].min(), test_df[x_feat].max()
  y_min, y_max = test_df[y_feat].min(), test_df[y_feat].max()
  x_range = np.linspace(
      x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1, 100
  )


  def add_lr_boundary(model, name, color, dash="dash"):
    try:
      idx_x = selected_features.index(x_feat)
      idx_y = selected_features.index(y_feat)
      w1 = model.coef_[0][idx_x]
      w2 = model.coef_[0][idx_y]
      intercept = model.intercept_[0]
      if abs(w2) > 1e-5:
        y_line = (-w1 * x_range - intercept) / w2
        in_bounds = (y_line >= y_min - (y_max - y_min) * 0.5) & (
            y_line <= y_max + (y_max - y_min) * 0.5
        )
        if not in_bounds.any():
          st.warning(f"⚠️ {name} 경계선이 현재 그림 범위 밖에 있습니다.")
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_line,
                mode="lines",
                name=name,
                line=dict(color=color, width=3, dash=dash),
            )
        )
    except Exception:
      pass


  add_lr_boundary(
      lr_before, "가중치 전 (Before) 0.5 경계선", "blue", "dash"
  )
  add_lr_boundary(lr_after, "가중치 후 (After) 0.5 경계선", "red", "solid")

  st.plotly_chart(fig, use_container_width=True)

elif len(selected_kor_features) >= 3:
  st.markdown(
      "📌 속성을 3개 이상 선택하셨으므로, 첫 세 개의 속성을 축으로 하는"
      " **3차원 산점도 및 평면 비교**를 표시합니다."
  )
  plot_kor_x, plot_kor_y, plot_kor_z = selected_kor_features[:3]
  x_feat, y_feat, z_feat = [kor_to_eng[f] for f in [plot_kor_x, plot_kor_y, plot_kor_z]]

  test_df["stroke_label"] = test_df["stroke"].map(
      {0: "뇌졸중 아님 (0)", 1: "뇌졸중 (1)"}
  )

  fig_3d = px.scatter_3d(
      test_df,
      x=x_feat,
      y=y_feat,
      z=z_feat,
      color="stroke_label",
      color_discrete_map={"뇌졸중 아님 (0)": "#1f77b4", "뇌졸중 (1)": "#ff7f0e"},
      title=f"3차원 산점도 및 가중치 전후 평면 비교 ({plot_kor_x}, {plot_kor_y}, {plot_kor_z})",
      opacity=0.8,
  )

  # 3차원 평면 격자 생성
  x_vals = np.linspace(test_df[x_feat].min(), test_df[x_feat].max(), 20)
  y_vals = np.linspace(test_df[y_feat].min(), test_df[y_feat].max(), 20)
  xx_3d, yy_3d = np.meshgrid(x_vals, y_vals)


  def add_3d_plane(model, name, colorscale):
    try:
      idx_x = selected_features.index(x_feat)
      idx_y = selected_features.index(y_feat)
      idx_z = selected_features.index(z_feat)
      w1 = model.coef_[0][idx_x]
      w2 = model.coef_[0][idx_y]
      w3 = model.coef_[0][idx_z]
      intercept = model.intercept_[0]

      if abs(w3) > 1e-5:
        zz_3d = (-w1 * xx_3d - w2 * yy_3d - intercept) / w3
        fig_3d.add_trace(
            go.Surface(
