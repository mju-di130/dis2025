import datetime
import logging
import os
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

# Optional: modeling libs
# 🌟 수정 1: SingleTableMetadata 임포트 로직을 다시 복구하고 load_from_dataframe 사용을 준비합니다.
SingleTableMetadata = None
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# 파일 및 경로 설정
INPUT_CSV = "data-utf8.csv"  # 파일명 확인(수정)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_TARGET = 5        # 목표 K-익명성
L_TARGET = 3        # 목표 L-다양성
EPSILON_DP = 0.5    # 차분프라이버시 ε (데모용)
SYNTH_ROWS = None   # None => same as original

# ---------- 유틸리티 함수 ----------
def load_data(path: str) -> pd.DataFrame:
    logging.info(f"Loading CSV: {path}")
    df = pd.read_csv(path, encoding='utf-8', quotechar='"', skipinitialspace=True)
    logging.info(f"Loaded {len(df)} rows, {len(df.columns)} cols")
    return df

def safe_to_numeric(series: pd.Series) -> pd.Series:
    """
    숫자 형 변환 시 쉼표 제거 등 전처리
    """
    s = series.astype(str).str.replace(",", "").str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors='coerce')

# ---------- 전처리 ----------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 날짜 변환
    for date_col in ["개찰일자", "최종낙찰일자"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    # 금액 숫자화
    for num_col in ["기초금액", "최종낙찰금액"]:
        if num_col in df.columns:
            df[num_col + "_num"] = safe_to_numeric(df[num_col])
    # 낙찰율 숫자화
    if "최종낙찰율" in df.columns:
        df["낙찰율_num"] = safe_to_numeric(df["최종낙찰율"])
    # 연구주제 컬럼(민감속성) 존재 확인
    if "입찰공고명" in df.columns and "연구주제" not in df.columns:
        df["연구주제"] = df["입찰공고명"].astype(str).str.slice(0, 60)
    return df

# ---------- K-익명성: 기관명 -> 상위분류 일반화 ----------
def infer_org_category(org_name: str) -> str:
    """
    규칙 기반 상위 분류 추출.
    """
    if pd.isna(org_name):
        return "UNKNOWN"
    s = str(org_name)
    keywords = {
        "방위사업청": "중앙행정기관",
        "국방": "중앙행정기관",
        "육군": "군",
        "해군": "군",
        "공군": "군",
        "대학교": "대학",
        "연구소": "연구기관",
        "연구원": "연구기관",
        "주식회사": "기업",
        "㈜": "기업",
        "사": "기관"
    }
    for k, v in keywords.items():
        if k in s:
            return v
    s_clean = s.replace(" ", "")
    if len(s_clean) > 12:
        return s_clean[:6]
    return s_clean

def apply_k_generalization(df: pd.DataFrame, org_col: str="공고기관명", new_col: str="기관_상위") -> pd.DataFrame:
    df = df.copy()
    df[new_col] = df[org_col].apply(infer_org_category) if org_col in df.columns else "UNKNOWN"
    return df

# ---------- L-다양성 검사 및 억제 ----------
def compute_l_diversity(df: pd.DataFrame, qi_cols: List[str], sensitive_col: str) -> pd.Series:
    """
    Returns per-equivalence-class distinct sensitive counts
    """
    groups = df.groupby(qi_cols)[sensitive_col].nunique()
    return groups

def enforce_l_diversity_suppression(df: pd.DataFrame, qi_cols: List[str], sensitive_col: str, L_target: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    If an equivalence class has distinct sensitive values < L_target,
    suppress the sensitive attribute (e.g., replace with 'SUPPRESSED').
    Returns (df_modified, report_df)
    """
    df = df.copy()
    uniq_counts = compute_l_diversity(df, qi_cols, sensitive_col)
    failing_ec = uniq_counts[uniq_counts < L_target].index.tolist()
    report = []
    
    df_copy = df.reset_index(drop=True)
    
    for ec_key in failing_ec:
        mask = True
        if isinstance(ec_key, tuple):
            for col_val, col_name in zip(ec_key, qi_cols):
                mask = mask & (df_copy[col_name] == col_val)
        else:
            mask = (df_copy[qi_cols[0]] == ec_key)
            
        current_rows = df_copy.loc[mask, :].shape[0]
        current_unique = df_copy.loc[mask, sensitive_col].nunique()
        
        report.append((str(ec_key), current_rows, current_unique))
        df_copy.loc[mask, sensitive_col] = "SUPPRESSED"
        
    df = df_copy
    report_df = pd.DataFrame(report, columns=["equivalence_key", "rows", "unique_sensitive_after"])
    return df, report_df

# ---------- 요약/저장 함수 ----------
def save_dataframe(df: pd.DataFrame, fname: str):
    path = os.path.join(OUTPUT_DIR, fname)
    df.to_csv(path, index=False, encoding='utf-8-sig') 
    logging.info(f"Saved: {path}")
    return path

def get_timestamp_str() -> str:
    # 타임스탬프를 파일명에 넣기
    now = datetime.datetime.now()
    timestamp_str = now.strftime('%Y-%m-%d_%H-%M-%S') # 원하는 형식으로 지정
    return timestamp_str

# ---------- 메인 파이프라인 ----------
def main():
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}. Please ensure the file is in the correct directory.")
        return

    # 1. 데이터 로드
    logging.info("1. CSV 데이터 로드 시작")
    df_raw = load_data(INPUT_CSV)
    df = preprocess(df_raw)

    # 1) K-익명성 전처리: 기관명 -> 기관_상위
    if "공고기관명" in df.columns:
        df = apply_k_generalization(df, org_col="공고기관명", new_col="기관_상위")
    else:
        df["기관_상위"] = "UNKNOWN"

    # quick K check
    qi_cols = ["기관_상위"]
    if "개찰일자" in df.columns:
        df["연도"] = df["개찰일자"].dt.year.fillna(-1).astype(int)
        qi_cols.append("연도")
        
    ec_sizes = df.groupby(qi_cols).size()
    logging.info(f"Equivalence class sizes: min={ec_sizes.min()}, median={ec_sizes.median()}, mean={ec_sizes.mean()}")

    if ec_sizes.min() < K_TARGET:
        logging.warning(f"Minimum EC size {ec_sizes.min()} < K_TARGET {K_TARGET}. Consider further generalization or suppression.")

    # 2) L-다양성 보정 (민감속성: 연구주제)
    sensitive_col = "연구주제"
    if sensitive_col not in df.columns:
        logging.warning(f"Sensitive column '{sensitive_col}' missing. Check input data or preprocess function.")
    else:
        df[sensitive_col] = df[sensitive_col].astype(str).fillna("NA") 
        ld = compute_l_diversity(df, qi_cols, sensitive_col)
        logging.info(f"L-diversity stats: min={ld.min()}, median={ld.median()}, mean={ld.mean()}")
        if ld.min() < L_TARGET:
            logging.info("Enforcing L-diversity by suppression for small ECs (demo approach).")
            df, l_report = enforce_l_diversity_suppression(df, qi_cols, sensitive_col, L_TARGET)
            # save
            save_dataframe(l_report, f'l_diversity_suppression_report_{get_timestamp_str()}.csv')
            logging.info("Suppression applied for ECs with low L-diversity. Sensitive values replaced by 'SUPPRESSED' in those ECs.")

    # Save preprocessed dataset
    preproc_path = save_dataframe(df, f'preprocessed_defense_rnd_{get_timestamp_str()}.csv')

    # ---------- 합성 데이터 생성 ----------

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
    GCsynthesizer = GaussianCopulaSynthesizer(metadata)

    GCsynthesizer.fit(df)

    # 5. 재현 정보 생성 (원본과 동일 건수)
    GCsynthesized_data = GCsynthesizer.sample(num_rows=len(df))

    # 6. 필드별 타입 복원 및 후처리 (필요시)
    GCsynthesized_data["최종낙찰일자"] = pd.to_datetime(GCsynthesized_data["최종낙찰일자"])

    # 7. 결과 저장
    output_path = os.path.join(OUTPUT_DIR, f'synthetic_data_GC_{get_timestamp_str()}.csv')
    GCsynthesized_data.to_csv(output_path, index=False, encoding='utf-8-sig')

    logging.info(f"재현 정보 생성 (GaussianCopulaSynthesizer)가 성공적으로 저장되었습니다: {output_path}")

    #--
    logging.info("합성기 초기화 및 학습 시작")
    CTGANsynthesizer = CTGANSynthesizer(metadata=metadata, epochs=10, cuda=True)

    CTGANsynthesizer.fit(df)

    # 5. 재현 정보 생성 (원본과 동일 건수)
    CTGANSynthesized_data = CTGANsynthesizer.sample(num_rows=len(df))

    # 6. 필드별 타입 복원 및 후처리 (필요시)
    CTGANSynthesized_data["최종낙찰일자"] = pd.to_datetime(CTGANSynthesized_data["최종낙찰일자"])

    # 7. 결과 저장
    output_path = os.path.join(OUTPUT_DIR, f'synthetic_data_CTGAN_{get_timestamp_str()}.csv')
    CTGANSynthesized_data.to_csv(output_path, index=False, encoding='utf-8-sig')

    logging.info(f"재현 정보 생성 (CTGANSynthesizer)가 성공적으로 저장되었습니다: {output_path}")

if __name__ == "__main__":
    main()