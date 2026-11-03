import numpy as np
import pandas as pd
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="모델 채점표 - 뇌졸중 예측 실습실", page_icon="📋", layout="wide"
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

st.title("📋 분류 모델 채점표 및 평가 실습실")
st.markdown("---")

# 1. 속성 고정 및 동일 데이터 전처리 / 분할
feature_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

# '분류 모델' 페이지와 동일하게 체질량지수를 제외한 4개 기본 선택
selected_kor_features = ["나이", "평균 혈당", "고혈압", "심장병"]
kor_to_eng = {v: k for k, v in feature_map.items()}
selected_features = [kor_to_eng[k] for k in selected_kor_features]

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

# 2. 모델 학습
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
dt_model.fit(X_train, y_train)

# 예측 수행
y_pred_lr = lr_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

# 기준 모델 (다수 클래스 예측)
majority_class = y_train.mode()[0]
y_pred_base = np.array([majority_class] * len(y_test))

# 테스트용 총 인원 및 뇌졸중인 사람 수
total_test_count = len(y_test)
actual_stroke_count = int(y_test.sum())

st.markdown(
    f"📌 **테스트 데이터 정보**: 총 **{total_test_count:,}명** 중 실제 뇌졸중 환자"
    f" 수는 **{actual_stroke_count:,}명**입니다."
)
st.markdown("---")


# 혼동행렬 지표 계산 함수 (TP, FP, FN, TN)
def get_confusion_matrix_counts(y_true, y_pred):
  # 양성(Positive) = 1 (뇌졸중), 음성(Negative) = 0 (아님)
  tp = int(((y_true == 1) & (y_pred == 1)).sum())
  fp = int(((y_true == 0) & (y_pred == 1)).sum())
  fn = int(((y_true == 1) & (y_pred == 0)).sum())
  tn = int(((y_true == 0) & (y_pred == 0)).sum())
  return tp, fp, fn, tn


# 성능 지표 계산 함수
def calculate_metrics(tp, fp, fn, tn, total):
  accuracy = (tp + tn) / total if total > 0 else 0
  # 재현율 분모: 실제 뇌졸중인 사람 수 (tp + fn)
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0
  # 정밀도 분모: 뇌졸중이라 예측한 사람 수 (tp + fp)
  predicted_positive = tp + fp
  if predicted_positive == 0:
    precision = None
    f1 = None
  else:
    precision = tp / predicted_positive
    if precision + recall == 0:
      f1 = 0
    else:
      f1 = (2 * precision * recall) / (precision + recall)
  return accuracy, recall, precision, f1


lr_tp, lr_fp, lr_fn, lr_tn = get_confusion_matrix_counts(y_test, y_pred_lr)
dt_tp, dt_fp, dt_fn, dt_tn = get_confusion_matrix_counts(y_test, y_pred_dt)
base_tp, base_fp, base_fn, base_tn = get_confusion_matrix_counts(
    y_test, y_pred_base
)

# 3. 혼동행렬 (2x2 표) 표시
st.subheader("1. 모델별 혼동행렬 (Confusion Matrix)")
st.markdown("행은 **실제 값**, 열은 **예측 값**을 나타냅니다.")

col_cm1, col_cm2, col_cm3 = st.columns(3)


def display_cm_table(title, tp, fp, fn, tn):
  cm_df = pd.DataFrame(
      [
          [
              f"TN: {tn:,}<br>(참 음성)",
              f"FP: {fp:,}<br>(거짓 양성)",
              f"{tn + fp:,}",
          ],
          [
              f"FN: {fn:,}<br>(거짓 음성)",
              f"TP: {tp:,}<br>(참 양성)",
              f"{fn + tp:,}",
          ],
          [f"{tn + fn:,}", f"{fp + tp:,}", f"{tn + fp + fn + tp:,}"],
      ],
      index=["실제: 아님(0)", "실제: 뇌졸중(1)", "합계"],
      columns=["예측: 아님(0)", "예측: 뇌졸중(1)", "합계"],
  )
  st.markdown(f"**{title}**")
  st.markdown(cm_df.to_html(escape=False, classes="table table-striped"), unsafe_allow_html=True)


with col_cm1:
  display_cm_table("로지스틱 회귀 (확률로 답하는 모델)", lr_tp, lr_fp, lr_fn, lr_tn)

with col_cm2:
  display_cm_table(
      "의사결정트리 (질문으로 답하는 모델)", dt_tp, dt_fp, dt_fn, dt_tn
  )

with col_cm3:
  display_cm_table("기준 모델 (다수 클래스 예측)", base_tp, base_fp, base_fn, base_tn)

st.markdown("---")

# 4. 평가지표 비교 표
st.subheader("2. 주요 평가지표 비교 (정확도, 재현율, 정밀도, F1 스코어)")

lr_acc, lr_rec, lr_prec, lr_f1 = calculate_metrics(
    lr_tp, lr_fp, lr_fn, lr_tn, total_test_count
)
dt_acc, dt_rec, dt_prec, dt_f1 = calculate_metrics(
    dt_tp, dt_fp, dt_fn, dt_tn, total_test_count
)
base_acc, base_rec, base_prec, base_f1 = calculate_metrics(
    base_tp, base_fp, base_fn, base_tn, total_test_count
)


def format_val(val, is_float=True):
  if val is None:
    return "계산할 수 없음"
  return f"{val * 100:.2f}%" if is_float else f"{val:.4f}"


metrics_data = {
    "평가 지표": ["정확도 (Accuracy)", "재현율 (Recall)", "정밀도 (Precision)", "F1 스코어"],
    "로지스틱 회귀 (확률로 답하는 모델)": [
        format_val(lr_acc),
        format_val(lr_rec),
        format_val(lr_prec),
        format_val(lr_f1, False) if lr_f1 is not None else "계산할 수 없음",
    ],
    "의사결정트리 (질문으로 답하는 모델)": [
        format_val(dt_acc),
        format_val(dt_rec),
        format_val(dt_prec),
        format_val(dt_f1, False) if dt_f1 is not None else "계산할 수 없음",
    ],
    "훈련용에서 많은 쪽으로만 답하는 모델": [
        format_val(base_acc),
        format_val(base_rec),
        format_val(base_prec),
        format_val(base_f1, False) if base_f1 is not None else "계산할 수 없음",
    ],
}

metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

st.markdown("---")

# 5. 오류 분석 목록 (놓친 환자 FN vs 헛짚은 사람 FP)
st.subheader("3. 오류 분석 목록 (놓친 환자 vs 헛짚은 사람)")

# 모델 선택 위젯
chosen_model_name = st.selectbox(
    "분석할 모델을 선택하세요:",
    ["로지스틱 회귀 (확률로 답하는 모델)", "의사결정트리 (질문으로 답하는 모델)"],
)

if "로지스틱 회귀" in chosen_model_name:
  selected_preds = y_pred_lr
  current_fn = lr_fn
  current_fp = lr_fp
else:
  selected_preds = y_pred_dt
  current_fn = dt_fn
  current_fp = dt_fp

# 테스트 데이터 프레임에 예측 결과 반영
eval_test_df = test_df.copy()
eval_test_df["pred"] = selected_preds

# 왼쪽 목록: 실제 뇌졸중(1)인데 아니라고(0) 예측한 사람 (FN - 놓친 환자)
fn_df = eval_test_df[(eval_test_df["stroke"] == 1) & (eval_test_df["pred"] == 0)].copy()
# 오른쪽 목록: 실제 아님(0)인데 뇌졸중(1)이라고 예측한 사람 (FP - 헛짚은 사람)
fp_df = eval_test_df[(eval_test_df["stroke"] == 0) & (eval_test_df["pred"] == 1)].copy()

# 나이가 많은 순서(내림차순) 정렬
fn_df = fn_df.sort_values(by="age", ascending=False).reset_index(drop=True)
fp_df = fp_df.sort_values(by="age", ascending=False).reset_index(drop=True)

# 20줄 넘으면 앞 20줄만 표시
fn_display = fn_df.head(20)
fp_display = fp_df.head(20)

col_list1, col_list2 = st.columns(2)

with col_list1:
  st.markdown(
      f"#### 🔍 놓친 환자 목록 (실제 뇌졸중 O, 예측 X) [총 {len(fn_df)}명]"
  )
  if len(fn_df) > 20:
    st.caption("목록이 20줄을 넘겨 상위 20줄만 표시합니다.")

  if not fn_display.empty:
    display_fn_table = pd.DataFrame({
        "번호 (ID)": fn_display.get("id", fn_display.index),
        "나이": fn_display["age"],
        "평균 혈당": fn_display["avg_glucose_level"],
        "고혈압": fn_display["hypertension"].map({0: "아님", 1: "있음"}),
        "심장병": fn_display["heart_disease"].map({0: "아님", 1: "있음"}),
    })
    st.dataframe(display_fn_table, use_container_width=True, hide_index=True)
  else:
    st.info("해당하는 인원이 없습니다.")

with col_list2:
  st.markdown(
      f"#### 🔍 헛짚은 사람 목록 (실제 뇌졸중 X, 예측 O) [총 {len(fp_df)}명]"
  )
  if len(fp_df) > 20:
    st.caption("목록이 20줄을 넘겨 상위 20줄만 표시합니다.")

  if not fp_display.empty:
    display_fp_table = pd.DataFrame({
        "번호 (ID)": fp_display.get("id", fp_display.index),
        "나이": fp_display["age"],
        "평균 혈당": fp_display["avg_glucose_level"],
        "고혈압": fp_display["hypertension"].map({0: "아님", 1: "있음"}),
        "심장병": fp_display["heart_disease"].map({0: "아님", 1: "있음"}),
    })
    st.dataframe(display_fp_table, use_container_width=True, hide_index=True)
  else:
    st.info("해당하는 인원이 없습니다.")

st.markdown("---")
st.markdown(
    f"💡 **확인**: 선택한 모델의 혼동행렬 상 **FN(거짓 음성) 수({current_fn}명)**는"
    f" 왼쪽 놓친 환자 목록의 총 인원 수와 일치하며, **FP(거짓 양성)"
    f" 수({current_fp}명)**는 오른쪽 헛짚은 사람 목록의 총 인원 수와"
    f" 일치합니다."
)
