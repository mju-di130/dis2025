# 시스템에 맞게 설치
# pip install pandas numpy matplotlib seaborn scikit-learn sdv diffprivlib
# sdv 설치 환경에 따라 추가 의존성(CTGAN 등)이 필요할 수 있음

"""
Defense dataset: Synthpop-like pipeline with K-anonymity preprocessing,
L-diversity handling, synth (SDV) generation, and simple DP noise addition.

Paths: set `INPUT_CSV` to your file path.
Outputs saved under ./outputs/
"""

import os
import json
import logging
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Optional: modeling libs
# 🌟 수정 1: SingleTableMetadata 임포트 로직을 다시 복구하고 load_from_dataframe 사용을 준비합니다.
SingleTableMetadata = None
try:
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    # 최신 경로 시도
    try:
        from sdv.metadata.single_table import SingleTableMetadata 
    except ImportError:
        # 구버전 경로 시도
        from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except Exception as e:
    logging.error(f"SDV import failed (Synthesizer or SingleTableMetadata not found): {e}")
    SDV_AVAILABLE = False

# DP library
try:
    from diffprivlib.mechanisms import LaplaceTruncated, Laplace
    DP_AVAILABLE = True
except Exception:
    DP_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 설정 ----------
INPUT_CSV = "data-utf8.cvs"  # 변경 가능
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_TARGET = 5        # 목표 K-익명성
L_TARGET = 3        # 목표 L-다양성
EPSILON_DP = 0.5    # 차분프라이버시 ε (데모용)
SYNTH_ROWS = None   # None => same as original

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- 유틸리티 함수 ----------
def load_data(path: str) -> pd.DataFrame:
    logging.info(f"Loading CSV: {path}")
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
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

# ---------- Synthpop-like 합성 (SDV 사용) ----------
def synthesize_with_sdv(df: pd.DataFrame, method: str="gaussiancopula", num_rows: int=None, random_state: int=0):
    """
    method: 'gaussiancopula' or 'ctgan'
    """
    global SingleTableMetadata

    if not SDV_AVAILABLE:
        raise ImportError("sdv 패키지가 필요합니다. pip install sdv")
    df = df.copy()
    if num_rows is None:
        num_rows = len(df)
    
    # 선택 컬럼: 제외할 컬럼 목록 정의
    exclude_cols = []
    for c in df.columns:
        if c.lower().find("대표자")>=0 or c.lower().find("담당자")>=0 or c.lower().find("주소")>=0:
            exclude_cols.append(c)
    if "공고기관명" in df.columns and "기관_상위" in df.columns:
        exclude_cols.append("공고기관명")
        
    cols_for_model = [c for c in df.columns if c not in exclude_cols]
    df_model = df[cols_for_model]
    
    metadata_obj = None
    if SingleTableMetadata:
        try:
            # 🌟 수정 2: load_from_dataframe을 사용하여 메타데이터 객체를 생성합니다.
            metadata_obj = SingleTableMetadata.load_from_dataframe(data=df_model)
            logging.info("Successfully created SingleTableMetadata object using load_from_dataframe.")
        except Exception as e:
            logging.error(f"Failed to create SingleTableMetadata object via load_from_dataframe (Error: {e}).")
            # 복구 전략으로 None을 사용하여 Synthesizer가 자체적으로 처리하도록 시도
            metadata_obj = None 
    
    # 만약 load_from_dataframe이 실패했거나 SingleTableMetadata가 임포트되지 않았다면,
    # metadata 인수를 아예 전달하지 않도록 fallback (이전 단계의 오류를 방지)
    
    model = None
    if method.lower() == "ctgan":
        if metadata_obj:
            # 🌟 수정 3: 메타데이터 객체가 성공적으로 생성되었을 경우에만 전달
            model = CTGANSynthesizer(metadata=metadata_obj, epochs=300, cuda=False)
        else:
            # 메타데이터 객체가 실패했을 경우, 인수를 제거하고 fit에 의존
            model = CTGANSynthesizer(epochs=300, cuda=False)
    else:
        if metadata_obj:
            # 🌟 수정 3: 메타데이터 객체가 성공적으로 생성되었을 경우에만 전달
            model = GaussianCopulaSynthesizer(metadata=metadata_obj) 
        else:
            # 메타데이터 객체가 실패했을 경우, 인수를 제거하고 fit에 의존
            model = GaussianCopulaSynthesizer()
        
    # Fit using selected columns.
    model.fit(df_model) # df_model 전달
    
    synth = model.sample(num_rows)
    
    # If excluded cols exist, we can attempt to map a fake token (or leave NaN)
    for c in exclude_cols:
        synth[c] = np.nan
        
    return synth

# ---------- 간단한 DP 노이즈 추가 (데모) ----------
def add_laplace_noise_column(values: pd.Series, epsilon: float, sensitivity: float=None, clip: Tuple[float,float]=None) -> pd.Series:
    """
    Adds Laplace noise with scale = sensitivity/epsilon.
    """
    if not DP_AVAILABLE:
        logging.error("diffprivlib 패키지가 없어 DP 노이즈를 추가할 수 없습니다. pip install diffprivlib")
        return values
        
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

# ---------- 메인 파이프라인 ----------
def main():
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}. Please ensure the file is in the correct directory.")
        return

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
            save_dataframe(l_report, "l_diversity_suppression_report.csv")
            logging.info("Suppression applied for ECs with low L-diversity. Sensitive values replaced by 'SUPPRESSED' in those ECs.")

    # Save preprocessed dataset
    preproc_path = save_dataframe(df, "preprocessed_defense_rnd.csv")

    # 3) synthpop-like synthetic generation using SDV
    if not SDV_AVAILABLE:
        logging.error("SDV not available: cannot synthesize. Install sdv package.")
        return

    rows = SYNTH_ROWS if SYNTH_ROWS is not None else len(df)
    logging.info(f"Generating synthetic data with rows={rows}. Using GaussianCopulaSynthesizer.")
    
    try:
        synth_df = synthesize_with_sdv(df, method="gaussiancopula", num_rows=rows)
    except Exception as e:
        if "FutureWarning" not in str(e):
             logging.error(f"Synthetic generation failed: {e}")
        return
        
    synth_path = save_dataframe(synth_df, "synth_synthpop_gaussiancopula.csv")

    # 4) DP noise addition to numeric columns (demo)
    num_cols = [c for c in synth_df.columns if synth_df[c].dtype.kind in 'fi']
    logging.info(f"Numeric cols for DP noise demo: {num_cols}")

    synth_dp = synth_df.copy()
    if DP_AVAILABLE:
        for col in num_cols:
            if col.endswith("_num") or col in ["최종낙찰율_num"]: 
                col_min, col_max = np.nanmin(synth_df[col]), np.nanmax(synth_df[col])
                sensitivity = float(col_max - col_min) if np.isfinite(col_max) and np.isfinite(col_min) else 1.0
                logging.info(f"Applying Laplace noise to {col}: sensitivity={sensitivity:.3f}, epsilon={EPSILON_DP}")
                synth_dp[col] = add_laplace_noise_column(synth_df[col], epsilon=EPSILON_DP, sensitivity=sensitivity)
    else:
        logging.warning("DP library not available, skipping noise addition.")

    dp_path = save_dataframe(synth_dp, f"synth_synthpop_gaussiancopula_dp_eps{EPSILON_DP}.csv")

    # 5) basic summaries and plots
    target_col = "최종낙찰금액_num"
    if target_col in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df[target_col].dropna(), bins=30, kde=True, label="orig", color="blue", alpha=0.5)
        sns.histplot(synth_df[target_col].dropna(), bins=30, kde=True, label="synth", color="orange", alpha=0.5)
        plt.legend(); plt.title("Final bid amount: original vs synthetic")
        plot_path = os.path.join(OUTPUT_DIR, "hist_final_bid_orig_vs_synth.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        logging.info(f"Saved plot: {plot_path}")
    else:
        logging.info(f"Target numeric column {target_col} not present for plot.")

    # Save metadata
    metadata_summary = {
        "input_file": INPUT_CSV,
        "rows_original": len(df_raw),
        "rows_synth": len(synth_df),
        "k_target": K_TARGET,
        "l_target": L_TARGET,
        "epsilon_dp": EPSILON_DP if DP_AVAILABLE else "N/A (DP not installed)",
        "sdv_used": SDV_AVAILABLE
    }
    with open(os.path.join(OUTPUT_DIR, "synthesis_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_summary, f, ensure_ascii=False, indent=2)
    logging.info("Pipeline finished.")

if __name__ == "__main__":
    main()