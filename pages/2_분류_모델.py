import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="분류 모델 실습 - 뇌졸중 예측 실습실", page_icon="🤖", layout="wide"
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
  st.error(f"데이터를 불러오는 중 오류가 하 발생했습니다: {e}")
  st.stop()

st.title("🤖 뇌졸중 예측 분류 모델 실습실")
st.markdown("---")

# 1. 입력 속성 선택 설정
st.subheader("1. 모델 입력 속성(Features) 선택")
st.markdown(
    "모델 학습에 사용할 속성을 선택해 주세요. (기본값: 체질량지수를 제외한"
    " 4개)"
)

# 속성 매핑 딕셔너리 (영어 열 이름 <-> 우리말 이름)
feature_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

selected_kor_features = st.multiselect(
    "사용할 속성을 고르세요 (2개 이상 선택):",
    options=list(feature_map.values()),
    default=["나이", "평균 혈당", "고혈압", "심장병"],
)

# 두 개보다 적게 고르면 안내문 표시 후 중단
if len(selected_kor_features) < 2:
  st.warning(
      "⚠️ 속성을 **2개 이상** 선택해 주세요. (가로축과 세로축 시각화 및"
      " 학습을 위해 최소 2개가 필요합니다.)"
  )
  st.stop()

# 선택한 우리말 이름을 영어 열 이름으로 변환
kor_to_eng = {v: k for k, v in feature_map.items()}
selected_features = [kor_to_eng[k] for k in selected_kor_features]

# 2. 전처리 및 데이터 분할 (지정된 조건)
# - bmi가 포함된 경우 훈련용의 중앙값으로 채움
# - 번호순 정렬 후 열 명씩 묶어 앞 세 명을 테스트용으로 고정 (나머지 칠 명으로 훈련)

# 데이터 복사본 생성 및 원본 인덱스(번호) 정렬
work_df = df.copy()
if "id" in work_df.columns:
  work_df = work_df.sort_values(by="id").reset_index(drop=True)
else:
  work_df = work_df.reset_index(drop=True)
  work_df.rename(columns={"index": "id"}, inplace=True)

test_indices = []
train_indices = []

# 열 명(10명)씩 묶어 앞 3명 테스트용, 나머지 7명 훈련용
for i in range(0, len(work_df), 10):
  group = work_df.iloc[i : i + 10]
  if len(group) >= 3:
    test_indices.extend(group.index[:3].tolist())
    train_indices.extend(group.index[3:].tolist())
  else:
    test_indices.extend(group.index.tolist())

train_df = work_df.loc[train_indices].copy()
test_df = work_df.loc[test_indices].copy()

# bmi 결측치 처리 (훈련용 데이터의 중앙값으로 훈련/테스트 각각 채우기)
if "bmi" in selected_features or "bmi" in work_df.columns:
  train_bmi_median = train_df["bmi"].median()
  train_df["bmi"] = train_df["bmi"].fillna(train_bmi_median)
  test_df["bmi"] = test_df["bmi"].fillna(train_bmi_median)

X_train = train_df[selected_features]
y_train = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

# 3. 모델 학습
# - 로지스틱 회귀 모델
# - 의사결정트리 모델 (max_depth=3, min_samples_leaf=5, random_state 고정)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
dt_model.fit(X_train, y_train)

# 정확도 계산
lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test))
lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))

dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test))
dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))

# 기준 모델 (Majority Classifier: 훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = y_train.mode()[0]
baseline_pred = [majority_class] * len(y_test)
baseline_test_acc = accuracy_score(y_test, baseline_pred)
baseline_train_pred = [majority_class] * len(y_train)
baseline_train_acc = accuracy_score(y_train, baseline_train_pred)

st.markdown("---")
st.subheader("2. 모델 성능 비교 (정확도)")

c1, c2, c3 = st.columns(3)

with c1:
  st.metric(
      label="로지스틱 회귀 (확률로 답하는 모델)",
      value=f"{lr_test_acc * 100:.2f}%",
      delta="테스트 정확도",
  )
  st.caption(
      f"• 훈련 정확도: {lr_train_acc * 100:.2f}%<br>• 테스트 정확도:"
      f" {lr_test_acc * 100:.2f}%",
      unsafe_allow_html=True,
  )

with c2:
  st.metric(
      label="의사결정트리 (질문으로 답하는 모델)",
      value=f"{dt_test_acc * 100:.2f}%",
      delta="테스트 정확도",
  )
  st.caption(
      f"• 훈련 정확도: {dt_train_acc * 100:.2f}%<br>• 테스트 정확도:"
      f" {dt_test_acc * 100:.2f}%",
      unsafe_allow_html=True,
  )

with c3:
  st.metric(
      label="기준 모델 (다수 클래스 예측)",
      value=f"{baseline_test_acc * 100:.2f}%",
      delta="테스트 정확도",
  )
  st.caption(
      f"• 훈련 정확도: {baseline_train_acc * 100:.2f}%<br>• 테스트 정확도:"
      f" {baseline_test_acc * 100:.2f}%",
      unsafe_allow_html=True,
  )

st.markdown("---")

# 4. 결정 경계 및 산점도 시각화
st.subheader("3. 2차원 결정 경계 및 테스트 데이터 산점도")
st.markdown(
    "선택한 속성 중 **정확히 2개**를 골라 가로축과 세로축으로 삼아 시각화합니다."
)

if len(selected_kor_features) >= 2:
  plot_kor_x = st.selectbox(
      "가로축 (X축) 속성 선택:",
      options=selected_kor_features,
      index=0,
  )
  remaining_opts = [f for f in selected_kor_features if f != plot_kor_x]
  plot_kor_y = st.selectbox(
      "세로축 (Y축) 속성 선택:",
      options=remaining_opts,
      index=0,
  )

  x_feat = kor_to_eng[plot_kor_x]
  y_feat = kor_to_eng[plot_kor_y]

  # 나머지 속성들은 테스트 데이터의 중앙값으로 고정
  other_feats = [f for f in selected_features if f not in [x_feat, y_feat]]
  fixed_vals = {}
  fixed_desc_list = []
  for feat in other_feats:
    med_val = test_df[feat].median()
    fixed_vals[feat] = med_val
    fixed_desc_list.append(f"**{feature_map[feat]}**: 중앙값 ({med_val:.2f})")

  if fixed_desc_list:
    st.info(
        "📌 두 축 외의 나머지 속성은 테스트 데이터의 중앙값으로 고정하여"
        f" 계산했습니다.<br>고정된 값: {', '.join(fixed_desc_list)}",
        unsafe_allow_html=True,
    )

  # 시각화 범위 그리드 생성
  x_min, x_max = test_df[x_feat].min(), test_df[x_feat].max()
  y_min, y_max = test_df[y_feat].min(), test_df[y_feat].max()
  x_range = np.linspace(x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1, 100)
  y_range = np.linspace(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1, 100)
  xx, yy = np.meshgrid(x_range, y_range)

  grid_df = pd.DataFrame(index=np.arange(xx.ravel().shape[0]))
  grid_df[x_feat] = xx.ravel()
  grid_df[y_feat] = yy.ravel()
  for feat in other_feats:
    grid_df[feat] = fixed_vals[feat]

  # 열 순서 맞추기
  grid_df = grid_df[selected_features]

  # 의사결정트리 예측 영역 계산 (배경 색상용)
  dt_grid_preds = dt_model.predict(grid_df)
  ZZ_dt = dt_grid_preds.reshape(xx.shape)

  # Plotly 산점도 및 결정 경계 작성
  test_df["stroke_label"] = test_df["stroke"].map(
      {0: "뇌졸중 아님 (0)", 1: "뇌졸중 (1)"}
  )

  fig = px.scatter(
      test_df,
      x=x_feat,
      y=y_feat,
      color="stroke_label",
      color_discrete_map={
          "뇌졸중 아님 (0): ": "#1f77b4",
          "뇌졸중 아님 (0)": "#1f77b4",
          "뇌졸중 (1)": "#ff7f0e",
      },
      labels={x_feat: plot_kor_x, y_feat: plot_kor_y},
      title=f"{plot_kor_x} vs {plot_kor_y} 결정 경계 및 테스트 데이터",
      opacity=0.8,
  )

  # 의사결정트리 영역 틴트 (Contour)
  fig.add_trace(
      go.Contour(
          z=ZZ_dt,
          x=x_range,
          y=y_range,
          showscale=False,
          colorscale=[[0, "rgba(31, 119, 180, 0.1)"], [1, "rgba(255, 127, 14, 0.1)"]],
          hoverinfo="skip",
          contours=dict(coloring="heatmap"),
      )
  )

  # 로지스틱 회귀 확률이 0.5가 되는 선(
