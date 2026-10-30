import json
import os
import pandas as pd
import streamlit as st

# 1. 페이지 설정 (브라우저 탭 제목 및 아이콘)
st.set_page_config(
    page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide"
)

# 2. 데이터 로드 함수 (캐싱 적용)
DATA_URL = "https://raw.githubusercontent.com/pqpqpoiqwe-creator/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
  df = pd.read_csv(DATA_URL, encoding="utf-8")
  return df


# 3. 데이터 로드 시도
try:
  df = load_data()
except Exception as e:
  st.error(
      f"데이터를 불러오는 중 오류가 발생했습니다. 인터넷 연결이나 URL을"
      f" 확인해주세요.\n오류 내용: {e}"
  )
  st.stop()

# 4. 저장 폴더 및 파일 경로 설정
SAVE_DIR = "내_기록"
os.makedirs(SAVE_DIR, exist_ok=True)
RECORD_FILE = os.path.join(SAVE_DIR, "stroke_app_records.json")


# 데이터 저장 함수
def save_records(meaning_dict, source_text):
  data_to_save = {"meanings": meaning_dict, "source_text": source_text}
  with open(RECORD_FILE, "w", encoding="utf-8") as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=4)


# 데이터 불러오기 함수
def load_records():
  if os.path.exists(RECORD_FILE):
    try:
      with open(RECORD_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      pass
  return {"meanings": {}, "source_text": ""}


saved_data = load_records()
saved_meanings = saved_data.get("meanings", {})
saved_source = saved_data.get("source_text", "")

# --- 화면 구성 ---

# 앱 제목
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("---")

# 첫 화면: 데이터 소개
st.subheader("📊 데이터셋 개요 및 소개")

# 핵심 통계 지표 (숫자 카드 4개)
total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = (stroke_count / total_people) * 100 if total_people > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
  st.metric(label="전체 사람 수", value=f"{total_people:,} 명")
with col2:
  st.metric(label="열(Column) 개수", value=f"{total_columns} 개")
with col3:
  st.metric(label="stroke가 1인 사람 수", value=f"{stroke_count:,} 명")
with col4:
  st.metric(label="stroke 환자 비율", value=f"{stroke_ratio:.2f}%")

st.markdown("<br>", unsafe_allow_html=True)

# 열 정보 테이블 구성 (열 이름, 우리말 뜻, 값의 종류, 빈 값 개수)
st.subheader("📝 데이터 열(Column) 구조 및 설명")
st.markdown(
    "교재를 참고하여 **우리말 뜻** 칸을 직접 채워 넣고 아래 저장 버튼을"
    " 눌러주세요."
)

column_info = []
for col in df.columns:
  col_type = str(df[col].dtype)
  null_count = int(df[col].isnull().sum())
  default_meaning = saved_meanings.get(col, "")
  column_info.append({
      "열 이름 (English)": col,
      "우리말 뜻": default_meaning,
      "값의 종류 (데이터 타입)": col_type,
      "빈 값(Missing) 개수": null_count,
  })

info_df = pd.DataFrame(column_info)

# 사용자가 편집할 수 있는 테이블 (Data Editor)
edited_info_df = st.data_editor(
    info_df,
    column_config={
        "열 이름 (English)": st.column_config.TextColumn(disabled=True),
        "값의 종류 (데이터 타입)": st.column_config.TextColumn(disabled=True),
        "빈 값(Missing) 개수": st.column_config.NumberColumn(disabled=True),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 보고 알맞은 우리말 뜻을 입력하세요.",
            required=False,
        ),
    },
    hide_index=True,
    use_container_width=True,
)

# 데이터 처음 다섯 줄 보여주기
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

# 데이터 출처 입력 칸
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📌 데이터 출처")
user_source_input = st.text_area(
    "데이터 출처 정보를 입력하세요 (교재에 있는 내용을 적어주세요):",
    value=saved_source,
    placeholder="예: 캐글(Kaggle) Stroke Prediction Dataset / 교재 출처 정보 등",
    height=100,
)

# 저장 버튼
if st.button("💾 변경 내용 저장하기", type="primary"):
  # 수정된 우리말 뜻을 딕셔너리로 변환
  new_meanings = {}
  for idx, row in edited_info_df.iterrows():
    new_meanings[row["열 이름 (English)"]] = row["우리말 뜻"]

  save_records(new_meanings, user_source_input)
  st.success("우리말 뜻과 출처 정보가 '내_기록' 폴더에 안전하게 저장되었습니다!")
  st.rerun()
