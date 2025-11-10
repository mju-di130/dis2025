"""
Defense dataset: Synthpop-like pipeline with K-anonymity preprocessing,
L-diversity handling, synth (SDV) generation, and simple DP noise addition.

이 스크립트는 원본 데이터를 로드하고, K-익명성 및 L-다양성 기법을 적용하여 익명화한 후,
SDV 모델을 사용하여 합성 데이터를 생성하고, 마지막으로 차분 프라이버시(DP) 노이즈를
추가한 후, 원본 및 합성 데이터의 통계적 유사성을 평가하는 그래프를 생성하는 전체 파이프라인입니다.

경로: `INPUT_CSV` 변수를 사용자의 파일 경로로 설정하세요.
출력: 모든 결과 파일(CSV, JSON, PNG)은 './outputs/' 디렉토리에 저장됩니다.
"""

import os
import json
import logging
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 선택적 라이브러리: SDV (합성 데이터 생성) 임포트
SingleTableMetadata = None
try:
    # SDV의 주요 합성 모델 임포트
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    try:
        # SDV v1.0 이상 버전의 메타데이터 임포트 시도
        from sdv.metadata.single_table import SingleTableMetadata 
    except ImportError:
        # SDV 이전 버전의 메타데이터 임포트 시도
        from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except Exception as e:
    # SDV 임포트 실패 시 로깅 및 플래그 설정
    logging.error(f"SDV import failed (Synthesizer or SingleTableMetadata not found): {e}")
    SDV_AVAILABLE = False

# 선택적 라이브러리: diffprivlib (차분 프라이버시) 임포트
try:
    from diffprivlib.mechanisms import LaplaceTruncated, Laplace
    DP_AVAILABLE = True
except Exception:
    # diffprivlib 임포트 실패 시 플래그 설정
    DP_AVAILABLE = False

# 시각화 라이브러리 임포트
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl 

# ---------- 한글 폰트 설정 함수 ----------
def set_korean_font():
    """운영체제별 주요 한글 폰트를 찾아 Matplotlib의 기본 폰트로 설정합니다."""
    font_paths = [
        # Windows용 맑은 고딕
        'C:/Windows/Fonts/malgun.ttf',
        # macOS용 애플 고딕
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        '/Library/Fonts/AppleGothic.ttf',
        # Linux용 나눔 고딕 (일반적)
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/nanum/NanumGothic.ttf'
    ]
    
    font_name = None
    for path in font_paths:
        if os.path.exists(path):
            # 폰트 경로를 찾으면 해당 폰트 이름을 설정
            font_name = fm.FontProperties(fname=path).get_name()
            break
            
    if font_name:
        # Matplotlib 기본 폰트 설정
        mpl.rc('font', family=font_name)
        # 마이너스 부호 깨짐 방지 설정
        mpl.rc('axes', unicode_minus=False)
        logging.info(f"Matplotlib font set to: {font_name}")
    else:
        logging.warning("No standard Korean font found. Korean text in plots may be broken.")
# -----------------------------------------------

# ---------- 전역 설정 ----------
INPUT_CSV = "data-utf8.cvs" # 입력 데이터 파일 경로
OUTPUT_DIR = "outputs"      # 출력 파일 저장 디렉토리
os.makedirs(OUTPUT_DIR, exist_ok=True) # 출력 디렉토리 생성 (이미 있다면 무시)

K_TARGET = 5        # K-익명성 목표 값 (동질 집합의 최소 레코드 수)
L_TARGET = 3        # L-다양성 목표 값 (민감 속성의 최소 다양성 수)
EPSILON_DP = 0.5    # 차분 프라이버시 (DP)의 엡실론 값 (작을수록 프라이버시 보호 강도가 높음)
SYNTH_ROWS = None   # 합성할 레코드 수 (None이면 원본 데이터와 동일한 수)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- 유틸리티 함수 ----------
def load_data(path: str) -> pd.DataFrame:
    """CSV 파일을 로드하고 기본 정보를 로깅합니다."""
    logging.info(f"Loading CSV: {path}")
    # CSV 로드 시 인코딩 및 메모리 최적화 옵션 적용
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    logging.info(f"Loaded {len(df)} rows, {len(df.columns)} cols")
    return df

def safe_to_numeric(series: pd.Series) -> pd.Series:
    """
    숫자 형 변환 시 쉼표 제거 등 전처리 및 오류 처리
    """
    # 문자열로 변환 후 쉼표(,) 제거 및 공백 제거
    s = series.astype(str).str.replace(",", "").str.strip()
    # 비어있는 문자열, 'nan', 'None' 등을 NaN(결측값)으로 대체
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    # 숫자형으로 변환 (변환할 수 없는 값은 NaN으로 처리)
    return pd.to_numeric(s, errors='coerce')

# ---------- 전처리 ----------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임의 초기 전처리 (날짜 변환, 금액 컬럼 숫자화)를 수행합니다.
    """
    df = df.copy()
    
    # 1. 날짜 변환: 개찰일자와 최종낙찰일자를 datetime 객체로 변환
    for date_col in ["개찰일자", "최종낙찰일자"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
    # 2. 금액 숫자화: 기초금액, 최종낙찰금액을 숫자로 변환하여 새 컬럼 생성
    for num_col in ["기초금액", "최종낙찰금액"]:
        if num_col in df.columns:
            df[num_col + "_num"] = safe_to_numeric(df[num_col])
            
    # 3. 낙찰율 숫자화: 최종낙찰율을 숫자로 변환하여 새 컬럼 생성
    if "최종낙찰율" in df.columns:
        df["낙찰율_num"] = safe_to_numeric(df["최종낙찰율"])
        
    # 4. 민감 속성(연구주제) 생성: '입찰공고명'의 일부를 추출하여 민감 속성으로 가정
    if "입찰공고명" in df.columns and "연구주제" not in df.columns:
        # 공고명의 앞 60자만 추출하여 민감 속성으로 지정
        df["연구주제"] = df["입찰공고명"].astype(str).str.slice(0, 60)
    
    return df

# ---------- K-익명성: 기관명 -> 상위분류 일반화 ----------
def infer_org_category(org_name: str) -> str:
    """기관명을 상위 범주로 일반화하는 로직 (K-익명성 준비)"""
    if pd.isna(org_name):
        return "UNKNOWN"
    s = str(org_name)
    
    # 주요 키워드를 기반으로 상위 범주 매핑
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
            
    # 위의 키워드에 해당하지 않으면, 이름의 일부를 잘라내어 일반화
    s_clean = s.replace(" ", "")
    if len(s_clean) > 12:
        return s_clean[:6] # 이름이 길면 앞 6글자만 사용
    return s_clean


def apply_k_generalization(df: pd.DataFrame, org_col: str="공고기관명", new_col: str="기관_상위") -> pd.DataFrame:
    """원본 기관명 컬럼에 일반화 함수를 적용하여 새 컬럼을 생성합니다."""
    df = df.copy()
    if org_col in df.columns:
        df[new_col] = df[org_col].apply(infer_org_category)
    else:
        df[new_col] = "UNKNOWN"
    return df

# ---------- L-다양성 검사 및 억제 ----------
def compute_l_diversity(df: pd.DataFrame, qi_cols: List[str], sensitive_col: str) -> pd.Series:
    """
    주어진 준식별자(QI) 컬럼 그룹별로 민감 속성(Sensitive)의 고유값 개수(L-다양성)를 계산합니다.
    """
    # 준식별자(qi_cols) 기준으로 그룹화하고 민감 속성의 고유값 개수(nunique)를 계산
    groups = df.groupby(qi_cols)[sensitive_col].nunique()
    return groups

def enforce_l_diversity_suppression(df: pd.DataFrame, qi_cols: List[str], sensitive_col: str, L_target: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    L-다양성 목표(L_target)를 충족하지 못하는 동질 집합의 민감 속성 값을 'SUPPRESSED'로 억제합니다.
    """
    df = df.copy()
    # L-다양성 값 계산
    uniq_counts = compute_l_diversity(df, qi_cols, sensitive_col)
    # L_target 미만인 동질 집합의 키(index) 추출
    failing_ec = uniq_counts[uniq_counts < L_target].index.tolist()
    report = [] # 억제된 집합에 대한 보고서
    
    df_copy = df.reset_index(drop=True)
    
    # L-다양성을 충족하지 못한 동질 집합에 대해 반복
    for ec_key in failing_ec:
        mask = True
        # 다중 컬럼 QI 키에 대한 마스크 생성
        if isinstance(ec_key, tuple):
            for col_val, col_name in zip(ec_key, qi_cols):
                mask = mask & (df_copy[col_name] == col_val)
        # 단일 컬럼 QI 키에 대한 마스크 생성
        else:
            mask = (df_copy[qi_cols[0]] == ec_key)
            
        # 해당 동질 집합의 레코드 수 및 기존 고유값 수 기록
        current_rows = df_copy.loc[mask, :].shape[0]
        current_unique = df_copy.loc[mask, sensitive_col].nunique()
        
        report.append((str(ec_key), current_rows, current_unique))
        # 해당 동질 집합의 민감 속성 값을 'SUPPRESSED'로 대체 (억제)
        df_copy.loc[mask, sensitive_col] = "SUPPRESSED"
        
    df = df_copy
    # 억제 보고서 데이터프레임 생성
    report_df = pd.DataFrame(report, columns=["equivalence_key", "rows", "unique_sensitive_before_suppression"])
    return df, report_df

# ---------- Synthpop-like 합성 (SDV 사용) ----------
def synthesize_with_sdv(df: pd.DataFrame, method: str="gaussiancopula", num_rows: int=None, random_state: int=0):
    """SDV 라이브러리를 사용하여 합성 데이터를 생성합니다."""
    global SingleTableMetadata

    if not SDV_AVAILABLE:
        raise ImportError("sdv 패키지가 필요합니다. pip install sdv")
        
    df = df.copy()
    if num_rows is None:
        num_rows = len(df)
    
    # 식별자로 간주되어 합성 모델 학습에서 제외할 컬럼 목록
    exclude_cols = []
    # 이름, 주소 등 민감한 문자열 정보가 포함된 컬럼 제외
    for c in df.columns:
        if c.lower().find("대표자")>=0 or c.lower().find("담당자")>=0 or c.lower().find("주소")>=0:
            exclude_cols.append(c)
    # 일반화 이전의 원본 기관명도 제외 (일반화된 '기관_상위'만 사용)
    if "공고기관명" in df.columns and "기관_상위" in df.columns:
        exclude_cols.append("공고기관명")
        
    # 합성 모델 학습에 사용할 컬럼 목록
    cols_for_model = [c for c in df.columns if c not in exclude_cols]
    df_model = df[cols_for_model]
    
    metadata_obj = None
    # 데이터프레임에서 자동으로 메타데이터를 추론하여 생성
    if SingleTableMetadata:
        try:
            metadata_obj = SingleTableMetadata.load_from_dataframe(data=df_model)
            logging.info("Successfully created SingleTableMetadata object using load_from_dataframe.")
        except Exception as e:
            logging.error(f"Failed to create SingleTableMetadata object via load_from_dataframe (Error: {e}).")
            metadata_obj = None 
    
    model = None
    # CTGAN 모델을 사용할 경우
    if method.lower() == "ctgan":
        # 메타데이터 객체가 있으면 이를 사용하여 모델 초기화
        if metadata_obj:
            model = CTGANSynthesizer(metadata=metadata_obj, epochs=300, cuda=False)
        else:
            model = CTGANSynthesizer(epochs=300, cuda=False)
    # Gaussian Copula 모델을 사용할 경우 (기본값)
    else:
        if metadata_obj:
            model = GaussianCopulaSynthesizer(metadata=metadata_obj) 
        else:
            model = GaussianCopulaSynthesizer()
        
    # 합성 모델 학습
    model.fit(df_model)
    
    # 학습된 모델로 합성 데이터 샘플 생성
    synth = model.sample(num_rows)
    
    # 학습에서 제외했던 컬럼들은 다시 추가하고 NaN(결측값)으로 채움
    for c in exclude_cols:
        synth[c] = np.nan
        
    return synth

# ---------- 간단한 DP 노이즈 추가 (데모) ----------
def add_laplace_noise_column(values: pd.Series, epsilon: float, sensitivity: float=None, clip: Tuple[float,float]=None) -> pd.Series:
    """
    Laplace 메커니즘을 사용하여 수치형 데이터에 차분 프라이버시(DP) 노이즈를 추가합니다.
    """
    if not DP_AVAILABLE:
        logging.error("diffprivlib 패키지가 없어 DP 노이즈를 추가할 수 없습니다.")
        return values
        
    # 결측값이 아닌 수치형 데이터만 복사하여 사용
    vals = values.astype(float).copy().dropna()
    
    if len(vals) == 0:
        return values
        
    # 클리핑(Clipping) 범위가 설정된 경우 적용
    if clip is not None:
        vals = vals.clip(clip[0], clip[1])
        
    # 민감도(Sensitivity)가 설정되지 않은 경우, 데이터 범위(최대값 - 최소값)를 사용
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if sensitivity is None:
        # 민감도는 최소 1.0 또는 데이터 범위 중 큰 값으로 설정
        sensitivity = float(max(1.0, vmax - vmin))
        
    # Laplace 분포의 스케일(scale) 계산: b = sensitivity / epsilon
    scale = sensitivity / float(epsilon)
    # Laplace 분포로부터 노이즈 생성
    noise = np.random.laplace(loc=0.0, scale=scale, size=len(vals))
    # 원본 값에 노이즈 추가
    noisy = vals + noise
    
    # 노이즈가 추가된 값을 원본 인덱스에 맞춰 Series로 재구성
    noisy_series = pd.Series(index=vals.index, data=noisy)
    # 기존 결측값은 유지하고, 숫자 값에만 노이즈를 추가한 값을 병합하여 반환
    return values.combine_first(noisy_series)

# ---------- 요약/저장 함수 ----------
def save_dataframe(df: pd.DataFrame, fname: str):
    """데이터프레임을 CSV 파일로 저장하고 경로를 반환합니다."""
    path = os.path.join(OUTPUT_DIR, fname)
    # 한글 깨짐 방지를 위해 'utf-8-sig' 인코딩 사용
    df.to_csv(path, index=False, encoding='utf-8-sig') 
    logging.info(f"Saved: {path}")
    return path

# ---------- 메인 파이프라인 ----------
def main():
    """전체 합성 데이터 생성 및 평가 파이프라인을 실행합니다."""
    # Matplotlib 한글 폰트 설정
    set_korean_font()

    # 입력 파일 존재 여부 확인
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}.")
        return

    # 1. 데이터 로드 및 전처리
    df_raw = load_data(INPUT_CSV)
    df = preprocess(df_raw)

    # 2. K-익명성 전처리: 기관명 -> 기관_상위 일반화
    if "공고기관명" in df.columns:
        df = apply_k_generalization(df, org_col="공고기관명", new_col="기관_상위")
    else:
        df["기관_상위"] = "UNKNOWN"

    # K-익명성 확인을 위한 준식별자(QI) 설정
    qi_cols = ["기관_상위"]
    if "개찰일자" in df.columns:
        # 연도 컬럼을 생성하여 준식별자에 추가
        df["연도"] = df["개찰일자"].dt.year.fillna(-1).astype(int)
        qi_cols.append("연도")
        
    # 동질 집합(Equivalence Class, EC)의 크기 계산 및 로깅
    ec_sizes = df.groupby(qi_cols).size()
    logging.info(f"Equivalence class sizes: min={ec_sizes.min()}, median={ec_sizes.median()}, mean={ec_sizes.mean()}")

    if ec_sizes.min() < K_TARGET:
        logging.warning(f"Minimum EC size {ec_sizes.min()} < K_TARGET {K_TARGET}. L-diversity suppression will be applied.")

    # 3. L-다양성 보정 (민감속성: 연구주제)
    sensitive_col = "연구주제"
    if sensitive_col not in df.columns:
        logging.warning(f"Sensitive column '{sensitive_col}' missing. Check input data or preprocess function.")
    else:
        # 민감 속성 결측값 처리
        df[sensitive_col] = df[sensitive_col].astype(str).fillna("NA") 
        # L-다양성 값 계산
        ld = compute_l_diversity(df, qi_cols, sensitive_col)
        logging.info(f"L-diversity stats: min={ld.min()}, median={ld.median()}, mean={ld.mean()}")
        
        # L_TARGET 미만인 경우 억제(Suppression) 적용
        if ld.min() < L_TARGET:
            logging.info(f"Enforcing L-diversity by suppression (L={L_TARGET}).")
            df, l_report = enforce_l_diversity_suppression(df, qi_cols, sensitive_col, L_TARGET)
            save_dataframe(l_report, "l_diversity_suppression_report.csv")
            logging.info("Suppression applied to sensitive values in low L-diversity ECs.")

    # 전처리 및 익명화된 데이터셋 저장
    preproc_path = save_dataframe(df, "preprocessed_defense_rnd.csv")

    # 4. SDV를 이용한 합성 데이터 생성
    if not SDV_AVAILABLE:
        logging.error("SDV not available: cannot synthesize.")
        return

    rows = SYNTH_ROWS if SYNTH_ROWS is not None else len(df)
    logging.info(f"Generating synthetic data with rows={rows}. Using GaussianCopulaSynthesizer.")
    
    try:
        # 합성 데이터 생성 실행 (기본 모델: GaussianCopula)
        synth_df = synthesize_with_sdv(df, method="gaussiancopula", num_rows=rows)
    except Exception as e:
        # 예외 처리
        if "FutureWarning" not in str(e):
             logging.error(f"Synthetic generation failed: {e}")
        return
        
    synth_path = save_dataframe(synth_df, "synth_synthpop_gaussiancopula.csv")

    # 5. DP 노이즈 추가 (수치형 컬럼에만 데모 적용)
    num_cols = [c for c in synth_df.columns if synth_df[c].dtype.kind in 'fi'] # 숫자형(float, int) 컬럼 목록
    logging.info(f"Numeric cols for DP noise demo: {num_cols}")

    synth_dp = synth_df.copy()
    if DP_AVAILABLE:
        for col in num_cols:
            # 금액 및 비율 관련 숫자형 컬럼에만 DP 노이즈 적용
            if col.endswith("_num") or col in ["최종낙찰율_num"]: 
                col_min, col_max = np.nanmin(synth_df[col]), np.nanmax(synth_df[col])
                # 민감도 계산 (데이터 범위)
                sensitivity = float(col_max - col_min) if np.isfinite(col_max) and np.isfinite(col_min) else 1.0
                logging.info(f"Applying Laplace noise to {col}: sensitivity={sensitivity:.3f}, epsilon={EPSILON_DP}")
                # DP 노이즈 추가 함수 호출
                synth_dp[col] = add_laplace_noise_column(synth_df[col], epsilon=EPSILON_DP, sensitivity=sensitivity)
    else:
        logging.warning("DP library not available, skipping noise addition.")

    dp_path = save_dataframe(synth_dp, f"synth_synthpop_gaussiancopula_dp_eps{EPSILON_DP}.csv")

    # 6. 통계적 유사성 평가 그래프 생성
    
    # A. 단변량 분포 비교: 수치형 히스토그램
    for target_col in ["최종낙찰금액_num", "기초금액_num"]:
        if target_col in df.columns and target_col in synth_df.columns:
            plt.figure(figsize=(8,5))
            
            # 이상치의 영향을 줄이기 위해 상위 99.5% 값을 기준으로 최대값 설정
            if df[target_col].dropna().empty:
                continue

            vmax = np.percentile(df[target_col].dropna(), 99.5)
            # 설정된 최대값으로 클리핑하고 결측값 제거
            df_plot = df[target_col].clip(upper=vmax).dropna()
            synth_plot = synth_df[target_col].clip(upper=vmax).dropna()

            # 원본 데이터 분포 (파란색)
            sns.histplot(df_plot, bins=30, kde=True, label="원본", color="blue", alpha=0.5, stat="density", common_norm=False)
            # 합성 데이터 분포 (주황색)
            sns.histplot(synth_plot, bins=30, kde=True, label="합성", color="orange", alpha=0.5, stat="density", common_norm=False)
            
            plt.legend(); plt.title(f"단변량 분포 비교: {target_col.replace('_num', '')} (Max={vmax:.2f})")
            plot_path = os.path.join(OUTPUT_DIR, f"hist_{target_col}_orig_vs_synth.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            logging.info(f"Saved plot: {plot_path}")
        else:
            logging.info(f"Numeric column {target_col} not present for plot.")

    # B. 단변량 분포 비교: 범주형 빈도 막대 그래프
    target_cat_col = "기관_상위"
    if target_cat_col in df.columns and target_cat_col in synth_df.columns:
        # 원본 데이터의 범주별 빈도(비율) 계산 및 출처 컬럼 추가
        df_orig_freq = df[target_cat_col].value_counts(normalize=True).reset_index()
        df_orig_freq['Source'] = '원본'
        # 합성 데이터의 범주별 빈도(비율) 계산 및 출처 컬럼 추가
        df_synth_freq = synth_df[target_cat_col].value_counts(normalize=True).reset_index()
        df_synth_freq['Source'] = '합성'
        
        # 컬럼 이름 통일
        df_orig_freq.columns = [target_cat_col, 'Frequency', 'Source']
        df_synth_freq.columns = [target_cat_col, 'Frequency', 'Source']
        
        # 두 데이터프레임 병합
        df_combined = pd.concat([df_orig_freq, df_synth_freq])

        plt.figure(figsize=(10, 6))
        # Seaborn barplot으로 원본과 합성 데이터를 나란히 비교 (hue='Source')
        sns.barplot(data=df_combined, x=target_cat_col, y='Frequency', hue='Source')
        plt.xticks(rotation=45, ha='right') # x축 레이블 회전
        plt.title(f"단변량 분포 비교: 범주형 빈도 ({target_cat_col})")
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"bar_{target_cat_col}_orig_vs_synth.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        logging.info(f"Saved plot: {plot_path}")
    else:
        logging.info(f"Categorical column {target_cat_col} not present for plot.")


    # C. 이변량 상관관계 비교: 산점도 (Scatter Plot)
    x_col, y_col = "최종낙찰금액_num", "낙찰율_num"
    if x_col in df.columns and y_col in df.columns:
        plt.figure(figsize=(12, 5))
        
        # 분포 시각화를 위해 x축 데이터의 이상치 제한
        x_vmax = np.percentile(df[x_col].dropna(), 99.5)
        
        # 1. 원본 데이터 산점도
        plt.subplot(1, 2, 1)
        # x축 클리핑, y축 결측값 제거 후 산점도 생성
        sns.scatterplot(x=df[x_col].clip(upper=x_vmax), y=df[y_col].dropna(), color="blue", alpha=0.6)
        plt.title(f"원본 데이터: {x_col.replace('_num', '')} vs {y_col.replace('_num', '')}")
        plt.xlabel(x_col.replace('_num', '')); plt.ylabel(y_col.replace('_num', ''))

        # 2. 합성 데이터 산점도
        plt.subplot(1, 2, 2)
        # x축 클리핑, y축 결측값 제거 후 산점도 생성
        sns.scatterplot(x=synth_df[x_col].clip(upper=x_vmax), y=synth_df[y_col].dropna(), color="orange", alpha=0.6)
        plt.title(f"합성 데이터: {x_col.replace('_num', '')} vs {y_col.replace('_num', '')}")
        plt.xlabel(x_col.replace('_num', '')); plt.ylabel(y_col.replace('_num', ''))
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"scatter_{x_col}_vs_{y_col}_orig_vs_synth.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        logging.info(f"Saved plot: {plot_path}")
    else:
        logging.info(f"Bivariate columns ({x_col}, {y_col}) not present for scatter plot.")

    # 7. 메타데이터 저장
    metadata_summary = {
        "input_file": INPUT_CSV,
        "rows_original": len(df_raw),
        "rows_synth": len(synth_df),
        "k_target": K_TARGET,
        "l_target": L_TARGET,
        "epsilon_dp": EPSILON_DP if DP_AVAILABLE else "N/A (DP not installed)",
        "sdv_used": SDV_AVAILABLE
    }
    # JSON 파일로 요약 정보 저장
    with open(os.path.join(OUTPUT_DIR, "synthesis_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_summary, f, ensure_ascii=False, indent=2)
    logging.info("Pipeline finished.")

if __name__ == "__main__":
    main()
