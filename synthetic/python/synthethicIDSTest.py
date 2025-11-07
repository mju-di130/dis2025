import os
import pandas as pd
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# 파일 및 경로 설정
INPUT_CSV = "data-utf8.csv"  # 파일명 확인(수정)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 데이터 로드
logging.info("CSV 데이터 로드 시작")
df = pd.read_csv(INPUT_CSV, encoding='utf-8', quotechar='"', skipinitialspace=True)

# 2. 컬럼명 확인 및 날짜 컬럼 변환
logging.info("컬럼명 확인 및 날짜 타입 변환")
expected_columns = [
    "입찰공고명", "공고기관명", "최종낙찰금액", "최종낙찰율", "최종낙찰일자",
    "최종낙찰업체명", "최종낙찰업체대표자명", "최종낙찰업체담당자명", "최종낙찰업체주소"
]
assert all(col in df.columns for col in expected_columns), "컬럼명이 일부 일치하지 않습니다."

df["최종낙찰일자"] = pd.to_datetime(df["최종낙찰일자"], errors='coerce')

# 3. SDV 메타데이터 정의 (필수 컬럼 타입 지정)
logging.info("SDV 메타데이터 정의")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

# '최종낙찰일자' 컬럼은 datetime 타입으로 지정
metadata.update_column("최종낙찰일자", sdtype="datetime")

# 4. 합성기(GaussianCopulaSynthesizer) 초기화 및 학습
logging.info("합성기 초기화 및 학습 시작")
synthesizer = GaussianCopulaSynthesizer(metadata)

synthesizer.fit(df)

# 5. 합성 데이터 생성 (원본과 동일 건수)
synthesized_data = synthesizer.sample(num_rows=len(df))

# 6. 필드별 타입 복원 및 후처리 (필요시)
synthesized_data["최종낙찰일자"] = pd.to_datetime(synthesized_data["최종낙찰일자"])

# 7. 결과 저장
output_path = os.path.join(OUTPUT_DIR, "synthetic_data.csv")
synthesized_data.to_csv(output_path, index=False, encoding='utf-8-sig')

logging.info(f"합성 데이터가 성공적으로 저장되었습니다: {output_path}")