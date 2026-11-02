import pandas as pd
import plotly.express as px
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="데이터 탐색 - 뇌졸중 예측 실습실", page_icon="📈", layout="wide"
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

st.title("📈 데이터 탐색 및 시각화")
st.markdown("---")

# 1. 나이와 평균 혈당의 분포 히스토그램
st.subheader("1. 나이(Age) 및 평균 혈당(Avg Glucose Level) 분포")
col_h1, col_h2 = st.columns(2)

with col_h1:
  fig_age = px.histogram(
      df, x="age", nbins=30, title="나이(Age) 분포", labels={"age": "나이"}
  )
  fig_age.update_layout(bargap=0.1)
  st.plotly_chart(fig_age, use_container_width=True)

with col_h2:
  fig_glucose = px.histogram(
      df,
      x="avg_glucose_level",
      nbins=30,
      title="평균 혈당(Avg Glucose Level) 분포",
      labels={"avg_glucose_level": "평균 혈당"},
  )
  fig_glucose.update_layout(bargap=0.1)
  st.plotly_chart(fig_glucose, use_container_width=True)

st.markdown("---")

# 2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교 (상자그림 및 평균값 표)
st.subheader("2. 뇌졸중 여부(Stroke)에 따른 나이 및 평균 혈당 비교")

df["stroke_label"] = df["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

col_b1, col_b2 = st.columns(2)

with col_b1:
  fig_box_age = px.box(
      df,
      x="stroke_label",
      y="age",
      color="stroke_label",
      title="뇌졸중 여부에 따른 나이 비교",
      labels={"stroke_label": "뇌졸중 여부", "age": "나이"},
  )
  st.plotly_chart(fig_box_age, use_container_width=True)

with col_b2:
  fig_box_glucose = px.box(
      df,
      x="stroke_label",
      y="avg_glucose_level",
      color="stroke_label",
      title="뇌졸중 여부에 따른 평균 혈당 비교",
      labels={
          "stroke_label": "뇌졸중 여부",
          "avg_glucose_level": "평균 혈당",
      },
  )
  st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표 제공
st.markdown("##### 📌 그룹별 평균값 요약 표")
mean_summary = (
    df.groupby("stroke_label")[["age", "avg_glucose_level"]]
    .mean()
    .reset_index()
)
mean_summary.columns = ["뇌졸중 여부", "평균 나이", "평균 혈당"]
st.dataframe(
    mean_summary.style.format(
        {"평균 나이": "{:.2f}세", "평균 혈당": "{:.2f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# 3. 고혈압 및 심장병 여부에 따른 뇌졸중 비율 막대그래프
st.subheader("3. 질환(고혈압, 심장병) 여부에 따른 뇌졸중 비율")

col_bar1, col_bar2 = st.columns(2)

with col_bar1:
  hypertension_stroke = (
      df.groupby("hypertension")["stroke"].mean().reset_index()
  )
  hypertension_stroke["hypertension_label"] = hypertension_stroke[
      "hypertension"
  ].map({0: "고혈압 없음", 1: "고혈압 있음"})
  hypertension_stroke["stroke_percent"] = (
      hypertension_stroke["stroke"] * 100
  )

  fig_hyp = px.bar(
      hypertension_stroke,
      x="hypertension_label",
      y="stroke_percent",
      text="stroke_percent",
      title="고혈압 여부에 따른 뇌졸중 비율",
      labels={
          "hypertension_label": "고혈압 여부",
          "stroke_percent": "뇌졸중 비율 (%)",
      },
  )
  fig_hyp.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
  fig_hyp.update_yaxes(
      range=[
          0,
          max(hypertension_stroke["stroke_percent"].max() * 1.3, 10),
      ]
  )
  st.plotly_chart(fig_hyp, use_container_width=True)

with col_bar2:
  heart_stroke = df.groupby("heart_disease")["stroke"].mean().reset_index()
  heart_stroke["heart_label"] = heart_stroke["heart_disease"].map(
      {0: "심장병 없음", 1: "심장병 있음"}
  )
  heart_stroke["stroke_percent"] = heart_stroke["stroke"] * 100

  fig_heart = px.bar(
      heart_stroke,
      x="heart_label",
      y="stroke_percent",
      text="stroke_percent",
      title="심장병 여부에 따른 뇌졸중 비율",
      labels={
          "heart_label": "심장병 여부",
          "stroke_percent": "뇌졸중 비율 (%)",
      },
  )
  fig_heart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
  fig_heart.update_yaxes(
      range=[0, max(heart_stroke["stroke_percent"].max() * 1.3, 10)]
  )
  st.plotly_chart(fig_heart, use_container_width=True)

st.markdown("---")

# 4. 체질량지수(bmi) 결측치 그룹과 전체 뇌졸중 비율 비교
st.subheader("4. 체질량지수(bmi) 결측치 그룹 vs 전체 뇌졸중 비율 비교")

total_stroke_rate = df["stroke"].mean() * 100

bmi_null_df = df[df["bmi"].isnull()]
if len(bmi_null_df) > 0:
  bmi_null_stroke_rate = bmi_null_df["stroke"].mean() * 100
else:
  bmi_null_stroke_rate = 0.0

comparison_data = pd.DataFrame({
    "구분": ["전체 데이터", "bmi 비어 있는 사람들"],
    "인원수": [len(df), len(bmi_null_df)],
    "뇌졸중 비율 (%)": [total_stroke_rate, bmi_null_stroke_rate],
})

st.dataframe(
    comparison_data.style.format({"뇌졸중 비율 (%)": "{:.2f}%"}),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# 5. 흡연 상태(smoking_status) 값별 인원수 표
st.subheader("5. 흡연 상태(Smoking Status)별 인원수")

smoking_counts = (
    df["smoking_status"]
    .fillna("Unknown (결측치)")
    .value_counts()
    .reset_index()
)
smoking_counts.columns = ["흡연 상태", "사람 수"]

st.dataframe(smoking_counts, use_container_width=True, hide_index=True)
