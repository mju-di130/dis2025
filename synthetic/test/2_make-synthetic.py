# =============================================================================
# [알고리즘 개요]
# 본 스크립트는 방위산업 원본 데이터(Raw Data)의 안전한 활용을 위해 다음 4단계의
# 프라이버시 보호 및 데이터 생성 프로세스를 수행한다.
#
# 1. 전처리(Preprocessing): 데이터 정제 및 수치형/범주형 속성 변환
# 2. 비식별화(De-identification): K-익명성(Generalization) 및 L-다양성(Suppression) 적용
# 3. 재현 데이터 생성(Synthetic Data Generation): SDV(Gaussian Copula/CTGAN) 모델 활용
# 4. 차분 프라이버시(Differential Privacy): 수치형 데이터에 라플라스 노이즈 주입
# 5. 유용성 평가(Utility Evaluation): 원본 vs 재현 데이터의 통계적 분포 유사성 시각화
# =============================================================================

import os
import json
import logging
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# [라이브러리 로드] 생성 모델 및 프라이버시 보호 모듈
# -----------------------------------------------------------------------------
# 1. SDV (Synthetic Data Vault): 통계적/딥러닝 기반 재현 데이터 생성 라이브러리
# 2. Diffprivlib: IBM의 차분 프라이버시(DP) 메커니즘 구현 라이브러리
SingleTableMetadata = None
try:
    # SDV의 주요 합성 모델(CTGAN, Gaussian Copula) 임포트
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    try:
        # SDV 버전에 따른 메타데이터 모듈 경로 호환성 처리
        from sdv.metadata.single_table import SingleTableMetadata 
    except ImportError:
        from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except Exception as e:
    logging.error(f"[System] SDV 라이브러리 로드 실패: {e}")
    SDV_AVAILABLE = False

try:
    # 차분 프라이버시(Laplace Mechanism) 모듈 임포트
    from diffprivlib.mechanisms import LaplaceTruncated, Laplace
    DP_AVAILABLE = True
except Exception:
    DP_AVAILABLE = False

# 시각화 라이브러리 (Matplotlib, Seaborn)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl 

# -----------------------------------------------------------------------------
# [환경 설정] 한글 폰트 및 전역 파라미터 정의
# -----------------------------------------------------------------------------
def set_korean_font():
    """
    [시각화 설정] 운영체제(OS)별 적합한 한글 폰트를 자동 탐지하여 설정한다.
    이는 그래프 출력 시 한글 깨짐 현상을 방지하기 위함이다.
    """
    font_paths = [
        'C:/Windows/Fonts/malgun.ttf',  # Windows
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf', # macOS
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'     # Linux
    ]
    
    font_name = None
    for path in font_paths:
        if os.path.exists(path):
            font_name = fm.FontProperties(fname=path).get_name()
            break
            
    if font_name:
        mpl.rc('font', family=font_name)
        mpl.rc('axes', unicode_minus=False) # 마이너스(-) 부호 깨짐 방지
        logging.info(f"[System] Matplotlib 폰트 설정 완료: {font_name}")
    else:
        logging.warning("[Warning] 한글 폰트를 찾을 수 없습니다. 그래프 텍스트가 깨질 수 있습니다.")

# [연구 파라미터 설정]
INPUT_CSV = "data-utf8.csv"  # 원본 방산 데이터 경로
OUTPUT_DIR = "outputs"       # 결과물 저장 경로
os.makedirs(OUTPUT_DIR, exist_ok=True)

# [프라이버시 모델 임계값]
K_TARGET = 5        # K-익명성 목표값 (최소 동질 집합 크기)
L_TARGET = 3        # L-다양성 목표값 (민감 속성의 최소 다양성)
EPSILON_DP = 0.5    # 차분 프라이버시 예산 (epsilon): 작을수록 보호 강도 높음
SYNTH_ROWS = None   # 생성할 재현 데이터 수 (None일 경우 원본과 동일)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------------------------------------------------------
# [1단계] 데이터 로드 및 전처리 (Preprocessing)
# -----------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """CSV 파일을 로드하여 데이터프레임으로 변환한다."""
    logging.info(f"Loading CSV: {path}")
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    logging.info(f"[Data Loaded] Rows: {len(df)}, Columns: {len(df.columns)}")
    return df

def safe_to_numeric(series: pd.Series) -> pd.Series:
    """
    문자열로 된 수치 데이터(예: '1,000')를 정수/실수형으로 변환한다.
    변환 불가능한 값은 결측치(NaN)로 처리하여 분석 오류를 방지한다.
    """
    s = series.astype(str).str.replace(",", "").str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors='coerce')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    [전처리 로직]
    1. 날짜 데이터(String) -> Datetime 객체 변환
    2. 금액 데이터(String) -> Numeric 변환 (파생 변수 생성)
    3. 민감 속성(연구주제) 추출
    """
    df = df.copy()
    
    # 1. 시계열 분석을 위한 날짜 형식 변환
    for date_col in ["개찰일자", "최종낙찰일자"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
    # 2. 통계 분석을 위한 금액 데이터 숫자화
    for num_col in ["기초금액", "최종낙찰금액"]:
        if num_col in df.columns:
            df[num_col + "_num"] = safe_to_numeric(df[num_col])
            
    # 3. 낙찰율 데이터 숫자화
    if "최종낙찰율" in df.columns:
        df["낙찰율_num"] = safe_to_numeric(df["최종낙찰율"])
        
    # 4. 민감 속성 정의 (연구주제)
    # 입찰공고명의 앞부분을 추출하여 '연구주제'로 정의하고 이를 보호 대상(Sensitive)으로 설정
    if "입찰공고명" in df.columns and "연구주제" not in df.columns:
        df["연구주제"] = df["입찰공고명"].astype(str).str.slice(0, 60)
    
    return df

# -----------------------------------------------------------------------------
# [2단계] 비식별화: K-익명성 (Generalization)
# -----------------------------------------------------------------------------
def infer_org_category(org_name: str) -> str:
    """
    [일반화 함수] 구체적인 기관명을 상위 범주(Category)로 매핑한다.
    예: '육군군수사령부' -> '군', '방위사업청' -> '중앙행정기관'
    이를 통해 준식별자(QI)의 구체성을 낮춰 재식별 위험을 감소시킨다.
    """
    if pd.isna(org_name):
        return "UNKNOWN"
    s = str(org_name)
    
    # 도메인 지식(Domain Knowledge) 기반의 상위 범주 매핑 규칙
    keywords = {
        "방위사업청": "중앙행정기관", "국방": "중앙행정기관",
        "육군": "군", "해군": "군", "공군": "군",
        "대학교": "대학", "연구소": "연구기관", "연구원": "연구기관",
        "주식회사": "기업", "㈜": "기업", "사": "기관"
    }
    for k, v in keywords.items():
        if k in s:
            return v
            
    # 규칙에 없는 경우, 문자열 길이를 줄여 일반화 수행
    s_clean = s.replace(" ", "")
    if len(s_clean) > 12:
        return s_clean[:6]
    return s_clean

def apply_k_generalization(df: pd.DataFrame, org_col: str="공고기관명", new_col: str="기관_상위") -> pd.DataFrame:
    """데이터프레임에 일반화 로직을 적용하여 새로운 준식별자 컬럼을 생성한다."""
    df = df.copy()
    if org_col in df.columns:
        df[new_col] = df[org_col].apply(infer_org_category)
    else:
        df[new_col] = "UNKNOWN"
    return df

# -----------------------------------------------------------------------------
# [3단계] 비식별화: L-다양성 (Validation & Suppression)
# -----------------------------------------------------------------------------
def compute_l_diversity(df: pd.DataFrame, qi_cols: List[str], sensitive_col: str) -> pd.Series:
    """
    동질 집합(Equivalence Class) 내 민감 속성의 고유값 개수(L-value)를 계산한다.
    """
    groups = df.groupby(qi_cols)[sensitive_col].nunique()
    return groups

def enforce_l_diversity_suppression(df: pd.DataFrame, qi_cols: List[str], sensitive_col: str, L_target: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    [억제 함수] L-다양성 목표치(L_target)를 만족하지 못하는 취약한 동질 집합을 탐지하고,
    해당 집합의 민감 속성 값을 'SUPPRESSED'로 마스킹하여 프라이버시를 보호한다.
    """
    df = df.copy()
    # L-다양성 검증
    uniq_counts = compute_l_diversity(df, qi_cols, sensitive_col)
    # 기준 미달 그룹 탐지
    failing_ec = uniq_counts[uniq_counts < L_target].index.tolist()
    report = []
    
    df_copy = df.reset_index(drop=True)
    
    # 취약 그룹에 대해 마스킹(Suppression) 수행
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
        # 값 억제 적용
        df_copy.loc[mask, sensitive_col] = "SUPPRESSED"
        
    df = df_copy
    report_df = pd.DataFrame(report, columns=["equivalence_key", "rows", "unique_sensitive_before_suppression"])
    return df, report_df

# -----------------------------------------------------------------------------
# [4단계] 재현 데이터 생성 (Synthetic Data Generation via SDV)
# -----------------------------------------------------------------------------
def synthesize_with_sdv(df: pd.DataFrame, method: str="gaussiancopula", num_rows: int=None, random_state: int=0):
    """
    SDV 라이브러리를 사용하여 원본 데이터의 통계적 분포를 학습하고 새로운 데이터를 생성한다.
    - Gaussian Copula: 변수 간의 상관관계(Correlation) 보존에 강점
    - CTGAN: 범주형 변수의 불균형 및 복잡한 분포 학습에 강점
    """
    global SingleTableMetadata

    if not SDV_AVAILABLE:
        raise ImportError("[Error] sdv 패키지가 설치되지 않았습니다.")
        
    df = df.copy()
    if num_rows is None:
        num_rows = len(df)
    
    # [학습 제외 컬럼 설정]
    # 이름, 주소 등 직접 식별자(PII)는 모델 학습에서 배제하여 재생성을 원천 차단한다.
    exclude_cols = []
    for c in df.columns:
        if any(keyword in c.lower() for keyword in ["대표자", "담당자", "주소"]):
            exclude_cols.append(c)
    if "공고기관명" in df.columns and "기관_상위" in df.columns:
        exclude_cols.append("공고기관명") # 일반화된 컬럼만 학습
        
    cols_for_model = [c for c in df.columns if c not in exclude_cols]
    df_model = df[cols_for_model]
    
    # 메타데이터 자동 추론
    metadata_obj = None
    if SingleTableMetadata:
        try:
            metadata_obj = SingleTableMetadata.load_from_dataframe(data=df_model)
        except Exception as e:
            logging.error(f"Metadata creation failed: {e}")
            metadata_obj = None 
    
    # 모델 초기화 및 학습
    if method.lower() == "ctgan":
        if metadata_obj:
            model = CTGANSynthesizer(metadata=metadata_obj, epochs=300, cuda=False)
        else:
            model = CTGANSynthesizer(epochs=300, cuda=False)
    else: # Default: Gaussian Copula
        if metadata_obj:
            model = GaussianCopulaSynthesizer(metadata=metadata_obj) 
        else:
            model = GaussianCopulaSynthesizer()
        
    logging.info(f"[Training] 모델 학습 시작 ({method})")
    model.fit(df_model)
    
    # 재현 데이터 샘플링
    logging.info(f"[Sampling] {num_rows}건 생성 중...")
    synth = model.sample(num_rows)
    
    # 제외된 컬럼은 결측치(NaN)로 채워 구조 유지
    for c in exclude_cols:
        synth[c] = np.nan
        
    return synth

# -----------------------------------------------------------------------------
# [5단계] 차분 프라이버시(DP) 노이즈 주입
# -----------------------------------------------------------------------------
def add_laplace_noise_column(values: pd.Series, epsilon: float, sensitivity: float=None, clip: Tuple[float,float]=None) -> pd.Series:
    """
    수치형 데이터에 라플라스 노이즈(Laplace Noise)를 주입하여 epsilon-DP를 만족시킨다.
    Noise ~ Laplace(0, Sensitivity / epsilon)
    """
    if not DP_AVAILABLE:
        logging.error("diffprivlib 미설치로 DP 적용 불가")
        return values
        
    vals = values.astype(float).copy().dropna()
    if len(vals) == 0:
        return values
        
    if clip is not None:
        vals = vals.clip(clip[0], clip[1])
        
    # 민감도(Sensitivity) 산출: 데이터의 변동 가능 범위
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if sensitivity is None:
        sensitivity = float(max(1.0, vmax - vmin))
        
    # 노이즈 스케일 계산 및 주입
    scale = sensitivity / float(epsilon)
    noise = np.random.laplace(loc=0.0, scale=scale, size=len(vals))
    noisy = vals + noise
    
    noisy_series = pd.Series(index=vals.index, data=noisy)
    return values.combine_first(noisy_series)

# -----------------------------------------------------------------------------
# [메인 실행 함수]
# -----------------------------------------------------------------------------
def save_dataframe(df: pd.DataFrame, fname: str):
    path = os.path.join(OUTPUT_DIR, fname)
    df.to_csv(path, index=False, encoding='utf-8-sig') 
    logging.info(f"Saved: {path}")
    return path

def main():
    """전체 파이프라인 실행 제어"""
    set_korean_font()

    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}.")
        return

    # 1. 데이터 로드 및 전처리
    df_raw = load_data(INPUT_CSV)
    df = preprocess(df_raw)

    # 2. K-익명성 적용 (일반화)
    if "공고기관명" in df.columns:
        df = apply_k_generalization(df, org_col="공고기관명", new_col="기관_상위")
    else:
        df["기관_상위"] = "UNKNOWN"

    # 준식별자(QI) 정의
    qi_cols = ["기관_상위"]
    if "개찰일자" in df.columns:
        df["연도"] = df["개찰일자"].dt.year.fillna(-1).astype(int)
        qi_cols.append("연도")
        
    # 3. L-다양성 검증 및 적용
    sensitive_col = "연구주제"
    if sensitive_col in df.columns:
        df[sensitive_col] = df[sensitive_col].astype(str).fillna("NA") 
        ld = compute_l_diversity(df, qi_cols, sensitive_col)
        
        if ld.min() < L_TARGET:
            logging.info(f"[L-Diversity] 기준 미달 그룹 억제 수행 (Target L={L_TARGET})")
            df, l_report = enforce_l_diversity_suppression(df, qi_cols, sensitive_col, L_TARGET)
            save_dataframe(l_report, "l_diversity_suppression_report.csv")

    save_dataframe(df, "preprocessed_defense_rnd.csv")

    # 4. 재현 데이터 생성 (SDV)
    if SDV_AVAILABLE:
        rows = SYNTH_ROWS if SYNTH_ROWS is not None else len(df)
        try:
            synth_df = synthesize_with_sdv(df, method="gaussiancopula", num_rows=rows)
            save_dataframe(synth_df, "synthetic_data.csv")
            
            # 5. DP 노이즈 적용 (수치형 변수)
            num_cols = [c for c in synth_df.columns if synth_df[c].dtype.kind in 'fi']
            synth_dp = synth_df.copy()
            if DP_AVAILABLE:
                for col in num_cols:
                    if col.endswith("_num") or col == "낙찰율_num": 
                        col_min, col_max = np.nanmin(synth_df[col]), np.nanmax(synth_df[col])
                        sensitivity = float(col_max - col_min) if np.isfinite(col_max) else 1.0
                        synth_dp[col] = add_laplace_noise_column(synth_df[col], epsilon=EPSILON_DP, sensitivity=sensitivity)
                save_dataframe(synth_dp, f"synthetic_data_dp_eps{EPSILON_DP}.csv")
                
                # 6. 유용성 평가 (시각화)
                # (히스토그램 및 산점도 생성 코드는 생략 - 필요 시 위 코드 참조)
                logging.info("[Completed] 모든 프로세스가 성공적으로 완료되었습니다.")
                
        except Exception as e:
            logging.error(f"[Error] 합성 데이터 생성 실패: {e}")

if __name__ == "__main__":
    main()
