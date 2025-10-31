import pandas as pd

# 파일 경로
file_path = "방위사업청_국내조달 경쟁 입찰결과_(2024년)-1.csv"

# 파일 불러오기
try:
    df = pd.read_csv(file_path, encoding="cp949", low_memory=False)
except:
    df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)

# 대상 컬럼명 정의
target_columns = [
    "입찰공고명", "공고기관명", "기초금액", "개찰일자", "최종낙찰금액", "최종낙찰율",
    "최종낙찰일자", "최종낙찰업체명", "최종낙찰업체대표자명", "최종낙찰업체담당자명", "최종낙찰업체주소"
]

# 실제 파일에 존재하는 컬럼만 필터링
existing_cols = [c for c in target_columns if c in df.columns]

# 연구개발 관련 키워드
keywords = ["연구", "개발", "R&D", "시험", "분석", "설계", "시제품", "모델", "시뮬레이션"]
pattern = "|".join(keywords)

# 필터링 수행
if "입찰공고명" in df.columns:
    rd_df = df[df["입찰공고명"].astype(str).str.contains(pattern, case=False, na=False)]
    rd_result = rd_df[existing_cols].head(30)  # 상위 30개
else:
    rd_result = pd.DataFrame({"오류": ["'입찰공고명' 열이 존재하지 않습니다."]})

rd_result
