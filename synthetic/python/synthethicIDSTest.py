import datetime
import logging
import json
import os
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

# Optional: modeling libs
# 🌟 수정 1: SingleTableMetadata 임포트 로직을 다시 복구하고 load_from_dataframe 사용을 준비합니다.
SingleTableMetadata = None
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

import matplotlib.pyplot as plt
import seaborn as sns

# 🌟 수정 1: 한글 폰트 설정을 위한 모듈 임포트
import matplotlib.font_manager as fm
import matplotlib as mpl 

# ---------- 한글 폰트 설정 함수 추가 ----------
def set_korean_font():
    """운영체제별 주요 한글 폰트를 찾아 Matplotlib의 기본 폰트로 설정합니다."""
    # 폰트 경로 목록 (운영체제별 주요 한글 폰트)
    font_paths = [
        # Windows
        'C:/Windows/Fonts/malgun.ttf',
        # macOS
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        '/Library/Fonts/AppleGothic.ttf', # 구 버전 macOS 경로
        # Linux (Nanum Gothic is common)
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/nanum/NanumGothic.ttf'
    ]
    
    font_name = None
    for path in font_paths:
        if os.path.exists(path):
            font_name = fm.FontProperties(fname=path).get_name()
            break
            
    if font_name:
        mpl.rc('font', family=font_name)
        # 마이너스 부호 깨짐 방지
        mpl.rc('axes', unicode_minus=False)
        logging.info(f"Matplotlib font set to: {font_name}")
    else:
        logging.warning("No standard Korean font found. Korean text in plots may be broken.")
# -----------------------------------------------

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

# ---------- 간단한 DP 노이즈 추가 (데모) ----------
def add_laplace_noise_column(values: pd.Series, epsilon: float, sensitivity: float=None, clip: Tuple[float,float]=None) -> pd.Series:
    """
    Adds Laplace noise with scale = sensitivity/epsilon.
    """

    vals = values.astype(float).copy().dropna()
    
    if len(vals) == 0:
        return values
        
    if clip is not None:
        vals = vals.clip(clip[0], clip[1])
        
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if sensitivity is None:
        sensitivity = float(max(1.0, vmax - vmin))
        
    scale = sensitivity / float(epsilon)
    noise = np.random.laplace(loc=0.0, scale=scale, size=len(vals))
    noisy = vals + noise
    
    noisy_series = pd.Series(index=vals.index, data=noisy)
    return values.combine_first(noisy_series)

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
    # 🌟 수정 4: 메인 함수 시작 시 한글 폰트 설정 함수 호출
    set_korean_font()

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

    # --
    # 4) DP noise addition to numeric columns (demo)
    num_cols = [c for c in CTGANSynthesized_data.columns if CTGANSynthesized_data[c].dtype.kind in 'fi']
    logging.info(f"Numeric cols for DP noise demo: {num_cols}")

    synth_dp = CTGANSynthesized_data.copy()
    for col in num_cols:
        if col.endswith("_num") or col in ["최종낙찰율_num"]: 
            col_min, col_max = np.nanmin(synth_dp[col]), np.nanmax(synth_dp[col])
            sensitivity = float(col_max - col_min) if np.isfinite(col_max) and np.isfinite(col_min) else 1.0
            logging.info(f"Applying Laplace noise to {col}: sensitivity={sensitivity:.3f}, epsilon={EPSILON_DP}")
            synth_dp[col] = add_laplace_noise_column(synth_dp[col], epsilon=EPSILON_DP, sensitivity=sensitivity)

    dp_path = save_dataframe(synth_dp, f"synth_synthpop_gaussiancopula_dp_eps{EPSILON_DP}_{get_timestamp_str()}.csv")

    # 5) basic summaries and plots

    # ----------------------------------------------------------------------
    ## 재현 정보 평가 그래프 추가 (Verification Plots)
    # ----------------------------------------------------------------------

    target_col = "최종낙찰금액_num"
    plt.figure(figsize=(8,5))
    sns.histplot(df[target_col].dropna(), bins=30, kde=True, label="orig", color="blue", alpha=0.5)
    sns.histplot(CTGANSynthesized_data[target_col].dropna(), bins=30, kde=True, label="synth", color="orange", alpha=0.5)
    plt.legend(); plt.title("Final bid amount: original vs synthetic")
    plot_path = os.path.join(OUTPUT_DIR, f'hist_final_bid_orig_vs_synth_{get_timestamp_str()}.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved plot: {plot_path}")

    # A. 단변량 분포 비교: 수치형 히스토그램
    for target_col in ["최종낙찰금액_num", "기초금액_num"]:
        if target_col in df.columns and target_col in CTGANSynthesized_data.columns:
            plt.figure(figsize=(8,5))
            
            # 💡 분포 시각화를 위해 값의 범위 조정 (Log 또는 표준화 대신, 현실적인 범위로 제한)
            # 상위 99% 값을 기준으로 최대값을 설정하여 이상치(outlier)의 영향을 줄입니다.
            if df[target_col].dropna().empty:
                continue

            vmax = np.percentile(df[target_col].dropna(), 99.5)
            df_plot = df[target_col].clip(upper=vmax).dropna()
            synth_plot = CTGANSynthesized_data[target_col].clip(upper=vmax).dropna()

            sns.histplot(df_plot, bins=30, kde=True, label="원본", color="blue", alpha=0.5, stat="density", common_norm=False)
            sns.histplot(synth_plot, bins=30, kde=True, label="합성", color="orange", alpha=0.5, stat="density", common_norm=False)
            
            plt.legend(); plt.title(f"단변량 분포 비교: {target_col.replace('_num', '')} (Max={vmax:.2f})")
            plot_path = os.path.join(OUTPUT_DIR, f"hist_{target_col}_orig_vs_synth_{get_timestamp_str()}.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            logging.info(f"Saved plot: {plot_path}")
        else:
            logging.info(f"Numeric column {target_col} not present for plot.")

    # B. 단변량 분포 비교: 범주형 빈도 막대 그래프
    target_cat_col = "기관_상위"
    if target_cat_col in df.columns and target_cat_col in CTGANSynthesized_data.columns:
        # 데이터프레임 병합 및 출처(source) 컬럼 생성
        df_orig_freq = df[target_cat_col].value_counts(normalize=True).reset_index()
        df_orig_freq['Source'] = '원본'
        df_synth_freq = CTGANSynthesized_data[target_cat_col].value_counts(normalize=True).reset_index()
        df_synth_freq['Source'] = '합성'
        
        # 컬럼 이름 통일 (SDV가 컬럼 이름을 변경하지 않았다고 가정)
        df_orig_freq.columns = [target_cat_col, 'Frequency', 'Source']
        df_synth_freq.columns = [target_cat_col, 'Frequency', 'Source']
        
        df_combined = pd.concat([df_orig_freq, df_synth_freq])

        plt.figure(figsize=(10, 6))
        # Seaborn barplot으로 원본과 합성 데이터를 나란히 비교
        sns.barplot(data=df_combined, x=target_cat_col, y='Frequency', hue='Source')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"단변량 분포 비교: 범주형 빈도 ({target_cat_col})")
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"bar_{target_cat_col}_orig_vs_synth_{get_timestamp_str()}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        logging.info(f"Saved plot: {plot_path}")
    else:
        logging.info(f"Categorical column {target_cat_col} not present for plot.")


    # C. 이변량 상관관계 비교: 산점도 (Scatter Plot)
    x_col, y_col = "최종낙찰금액_num", "낙찰율_num"
    if x_col in df.columns and y_col in df.columns:
        plt.figure(figsize=(12, 5))
        
        # 💡 분포 시각화를 위해 값의 범위 조정 (이상치 제한)
        x_vmax = np.percentile(df[x_col].dropna(), 99.5)
        
        # 원본 데이터 산점도
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=df[x_col].clip(upper=x_vmax), y=df[y_col].dropna(), color="blue", alpha=0.6)
        plt.title(f"원본 데이터: {x_col.replace('_num', '')} vs {y_col.replace('_num', '')}")
        plt.xlabel(x_col.replace('_num', '')); plt.ylabel(y_col.replace('_num', ''))

        # 합성 데이터 산점도
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=CTGANSynthesized_data[x_col].clip(upper=x_vmax), y=CTGANSynthesized_data[y_col].dropna(), color="orange", alpha=0.6)
        plt.title(f"합성 데이터: {x_col.replace('_num', '')} vs {y_col.replace('_num', '')}")
        plt.xlabel(x_col.replace('_num', '')); plt.ylabel(y_col.replace('_num', ''))
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"scatter_{x_col}_vs_{y_col}_orig_vs_synth_{get_timestamp_str()}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        logging.info(f"Saved plot: {plot_path}")
    else:
        logging.info(f"Bivariate columns ({x_col}, {y_col}) not present for scatter plot.")

    # ----------------------------------------------------------------------

    # Save metadata
    metadata_summary = {
        "input_file": INPUT_CSV,
        "rows_original": len(df_raw),
        "rows_synth": len(CTGANSynthesized_data),
        "k_target": K_TARGET,
        "l_target": L_TARGET,
        "epsilon_dp": EPSILON_DP if True else "N/A (DP not installed)",
        "sdv_used": True
    }
    with open(os.path.join(OUTPUT_DIR, f'synthesis_metadata_{get_timestamp_str()}.json'), "w", encoding="utf-8") as f:
        json.dump(metadata_summary, f, ensure_ascii=False, indent=2)
    logging.info("Pipeline finished.")

if __name__ == "__main__":
    main()