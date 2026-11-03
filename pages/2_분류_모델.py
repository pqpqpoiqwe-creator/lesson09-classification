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
  st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
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
# - 번호순 정렬 후 열 명씩 묶어 앞 세 명 테스트용 고정 (나머지 칠 명 훈련)

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

# bmi 결측치 처리 (훈련용 데이터의 중앙값으로 채우기)
if "bmi" in selected_features or "bmi" in work_df.columns:
  train_bmi_median = train_df["bmi"].median()
  train_
