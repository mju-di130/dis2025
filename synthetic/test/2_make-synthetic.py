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
#
# [출력 설정]
# - 모든 결과 파일(.csv, .png, .json)은 스크립트가 실행되는 '현재 디렉토리'에 저장된다.
# - 파일명 및 로그에서 가변적인 시간 정보(Timestamp)를 배제하여 재현성을 확보한다.
#
# [결과물 요약]
# 연번 / 구분 / 파일명 / 설명
# 1. 전처리 데이터 / preprocessed_defense_rnd.csv / 원본 데이터에 전처리(날짜/금액 변환)와 K-익명성(기관명 일반화)이 적용된 데이터
# 2. L-다양성 보고서 / l_diversity_suppression_report.csv / L-다양성 기준(L=3)을 만족하지 못해 마스킹(Suppression) 처리된 그룹의 정보
# 3. 재현 데이터 (Raw) / synthetic_data.csv / SDV 모델(Gaussian Copula 등)을 통해 생성된 초기 재현 데이터
# 4. 재현 데이터 (Final) / synthetic_data_dp_eps0.5.csv / 수치형 변수에 차분 프라이버시(DP) 노이즈(ϵ=0.5)까지 최종 적용된 데이터
# 5. 메타데이터 / synthesis_metadata.json / 실험에 사용된 파라미터(K,L,ϵ) 및 데이터 건수 요약 정보
# 6. 이미지 (분포) / hist_최종낙찰금액_num_orig_vs_synth.png / 원본 vs 재현 데이터의 금액 분포 비교 히스토그램
# 7. 이미지 (상관관계) / scatter_최종낙찰금액_num_vs_낙찰율_num.png / 금액과 낙찰율 간의 상관관계를 보여주는 산점도 비교 그래프
#
# [수정 사항]
# - 생성 모델(CTGAN vs Gaussian Copula) 선택에 따라 출력 파일명에 모델명을 자동 표기하도록 개선
# - 예: synthetic_data_ctgan.csv, synthetic_data_gaussiancopula.csv
# =============================================================================

import os
import json
import logging
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl 

# -----------------------------------------------------------------------------
# [설정] 생성 모델 선택 (이 부분을 변경하세요)
# -----------------------------------------------------------------------------
# 옵션: "ctgan" 또는 "gaussiancopula"
# MODEL_METHOD = "ctgan" 
MODEL_METHOD = "gaussiancopula" # None

# -----------------------------------------------------------------------------
# [라이브러리 로드]
# -----------------------------------------------------------------------------
SingleTableMetadata = None
try:
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    try:
        from sdv.metadata.single_table import SingleTableMetadata 
    except ImportError:
        from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except Exception as e:
    SDV_AVAILABLE = False

try:
    from diffprivlib.mechanisms import Laplace
    DP_AVAILABLE = True
except Exception:
    DP_AVAILABLE = False

# -----------------------------------------------------------------------------
# [환경 설정]
# -----------------------------------------------------------------------------
def set_korean_font():
    font_paths = [
        'C:/Windows/Fonts/malgun.ttf', 
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    ]
    font_name = None
    for path in font_paths:
        if os.path.exists(path):
            font_name = fm.FontProperties(fname=path).get_name()
            break
    if font_name:
        mpl.rc('font', family=font_name)
        mpl.rc('axes', unicode_minus=False)

# [연구 파라미터]
INPUT_CSV = "data-utf8.csv"
OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_TARGET = 5
L_TARGET = 3
EPSILON_DP = 0.5
SYNTH_ROWS = None

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# -----------------------------------------------------------------------------
# [1단계] 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    logging.info(f"Loading CSV: {path}")
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    return df

def safe_to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "").str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors='coerce')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for date_col in ["개찰일자", "최종낙찰일자"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    for num_col in ["기초금액", "최종낙찰금액"]:
        if num_col in df.columns:
            df[num_col + "_num"] = safe_to_numeric(df[num_col])
    if "최종낙찰율" in df.columns:
        df["낙찰율_num"] = safe_to_numeric(df["최종낙찰율"])
    if "입찰공고명" in df.columns and "연구주제" not in df.columns:
        df["연구주제"] = df["입찰공고명"].astype(str).str.slice(0, 60)
    return df

# -----------------------------------------------------------------------------
# [2단계] 비식별화: K-익명성
# -----------------------------------------------------------------------------
def infer_org_category(org_name: str) -> str:
    if pd.isna(org_name): return "UNKNOWN"
    s = str(org_name)
    keywords = {"방위사업청": "중앙행정기관", "국방": "중앙행정기관", "육군": "군", "해군": "군", "공군": "군", "대학교": "대학", "연구소": "연구기관", "연구원": "연구기관", "주식회사": "기업", "㈜": "기업", "사": "기관"}
    for k, v in keywords.items():
        if k in s: return v
    s_clean = s.replace(" ", "")
    if len(s_clean) > 12: return s_clean[:6]
    return s_clean

def apply_k_generalization(df: pd.DataFrame, org_col: str="공고기관명", new_col: str="기관_상위") -> pd.DataFrame:
    df = df.copy()
    if org_col in df.columns:
        df[new_col] = df[org_col].apply(infer_org_category)
    else:
        df[new_col] = "UNKNOWN"
    return df

# -----------------------------------------------------------------------------
# [3단계] 비식별화: L-다양성
# -----------------------------------------------------------------------------
def compute_l_diversity(df: pd.DataFrame, qi_cols: List[str], sensitive_col: str) -> pd.Series:
    groups = df.groupby(qi_cols)[sensitive_col].nunique()
    return groups

def enforce_l_diversity_suppression(df: pd.DataFrame, qi_cols: List[str], sensitive_col: str, L_target: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    report_df = pd.DataFrame(report, columns=["equivalence_key", "rows", "unique_sensitive_before_suppression"])
    return df, report_df

# -----------------------------------------------------------------------------
# [4단계] 재현 데이터 생성 (SDV) - 모델 선택 적용
# -----------------------------------------------------------------------------
def synthesize_with_sdv(df: pd.DataFrame, method: str="gaussiancopula", num_rows: int=None):
    if not SDV_AVAILABLE:
        raise ImportError("[Error] sdv 패키지가 설치되지 않았습니다.")
        
    df = df.copy()
    if num_rows is None: num_rows = len(df)
    
    exclude_cols = []
    for c in df.columns:
        if any(keyword in c.lower() for keyword in ["대표자", "담당자", "주소"]):
            exclude_cols.append(c)
    if "공고기관명" in df.columns and "기관_상위" in df.columns:
        exclude_cols.append("공고기관명") 
        
    cols_for_model = [c for c in df.columns if c not in exclude_cols]
    df_model = df[cols_for_model]
    
    # 메타데이터 자동 추론
    logging.info("[Metadata] 메타데이터 자동 추론 중...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_model)
    
    # 모델 선택 및 초기화
    if method.lower() == "ctgan":
        logging.info(f"[Model] CTGAN 모델 선택됨 (Epochs=300)")
        model = CTGANSynthesizer(metadata=metadata, epochs=300, cuda=False)
    else: 
        logging.info(f"[Model] Gaussian Copula 모델 선택됨")
        model = GaussianCopulaSynthesizer(metadata=metadata) 
        
    logging.info(f"[Training] 모델 학습 시작...")
    model.fit(df_model)
    
    logging.info(f"[Sampling] {num_rows}건 생성 중...")
    synth = model.sample(num_rows)
    
    for c in exclude_cols: synth[c] = np.nan
    return synth

# -----------------------------------------------------------------------------
# [5단계] 차분 프라이버시(DP) 노이즈 주입
# -----------------------------------------------------------------------------
def add_laplace_noise_column(values: pd.Series, epsilon: float, sensitivity: float=None, clip: Tuple[float,float]=None) -> pd.Series:
    if not DP_AVAILABLE:
        logging.error("diffprivlib 미설치로 DP 적용 불가")
        return values
    vals = values.astype(float).copy().dropna()
    if len(vals) == 0: return values
    if clip is not None: vals = vals.clip(clip[0], clip[1])
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if sensitivity is None: sensitivity = float(max(1.0, vmax - vmin))
    scale = sensitivity / float(epsilon)
    noise = np.random.laplace(loc=0.0, scale=scale, size=len(vals))
    noisy = vals + noise
    noisy_series = pd.Series(index=vals.index, data=noisy)
    return values.combine_first(noisy_series)

# -----------------------------------------------------------------------------
# [유틸리티] 결과 저장 함수
# -----------------------------------------------------------------------------
def save_dataframe(df: pd.DataFrame, fname: str):
    path = os.path.join(OUTPUT_DIR, fname)
    df.to_csv(path, index=False, encoding='utf-8-sig') 
    logging.info(f"Saved: {path}")
    return path

# -----------------------------------------------------------------------------
# [메인 실행 함수]
# -----------------------------------------------------------------------------
def main():
    set_korean_font()

    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}.")
        return

    # 1. 데이터 로드 및 전처리
    df_raw = load_data(INPUT_CSV)
    df = preprocess(df_raw)

    # 2. K-익명성
    if "공고기관명" in df.columns:
        df = apply_k_generalization(df, org_col="공고기관명", new_col="기관_상위")
    else:
        df["기관_상위"] = "UNKNOWN"

    qi_cols = ["기관_상위"]
    if "개찰일자" in df.columns:
        df["연도"] = df["개찰일자"].dt.year.fillna(-1).astype(int)
        qi_cols.append("연도")
        
    # 3. L-다양성
    sensitive_col = "연구주제"
    if sensitive_col in df.columns:
        df[sensitive_col] = df[sensitive_col].astype(str).fillna("NA") 
        ld = compute_l_diversity(df, qi_cols, sensitive_col)
        if ld.min() < L_TARGET:
            df, l_report = enforce_l_diversity_suppression(df, qi_cols, sensitive_col, L_TARGET)
            save_dataframe(l_report, "l_diversity_suppression_report.csv")

    save_dataframe(df, "preprocessed_defense_rnd.csv")

    # 4. 재현 데이터 생성
    if SDV_AVAILABLE:
        rows = SYNTH_ROWS if SYNTH_ROWS is not None else len(df)
        try:
            # [수정] MODEL_METHOD 변수에 따라 함수 호출
            synth_df = synthesize_with_sdv(df, method=MODEL_METHOD, num_rows=rows)
            
            # [수정] 파일명에 모델명(MODEL_METHOD) 포함
            save_dataframe(synth_df, f"synthetic_data_{MODEL_METHOD}.csv")
            
            # 5. DP 노이즈 적용
            num_cols = [c for c in synth_df.columns if synth_df[c].dtype.kind in 'fi']
            synth_dp = synth_df.copy()
            if DP_AVAILABLE:
                for col in num_cols:
                    if col.endswith("_num") or col == "낙찰율_num": 
                        col_min, col_max = np.nanmin(synth_df[col]), np.nanmax(synth_df[col])
                        sensitivity = float(col_max - col_min) if np.isfinite(col_max) else 1.0
                        synth_dp[col] = add_laplace_noise_column(synth_df[col], epsilon=EPSILON_DP, sensitivity=sensitivity)
                
                # [수정] 파일명에 모델명 포함
                save_dataframe(synth_dp, f"synthetic_data_{MODEL_METHOD}_dp_eps{EPSILON_DP}.csv")
                
                # 6. 유용성 평가 시각화
                # (1) 히스토그램
                for target_col in ["최종낙찰금액_num", "기초금액_num"]:
                    if target_col in df.columns and target_col in synth_dp.columns:
                        plt.figure(figsize=(8,5))
                        vmax = np.percentile(df[target_col].dropna(), 99.5)
                        sns.histplot(df[target_col].clip(upper=vmax).dropna(), bins=30, kde=True, label="원본", color="blue", alpha=0.5, stat="density", common_norm=False)
                        sns.histplot(synth_dp[target_col].clip(upper=vmax).dropna(), bins=30, kde=True, label=f"재현({MODEL_METHOD})", color="orange", alpha=0.5, stat="density", common_norm=False)
                        plt.legend()
                        plt.title(f"분포 비교 ({MODEL_METHOD}): {target_col.replace('_num', '')}")
                        
                        # [수정] 파일명에 모델명 포함
                        plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{target_col}_orig_vs_{MODEL_METHOD}.png"), bbox_inches='tight', dpi=150)
                        plt.close()

                # (2) 산점도
                x_col, y_col = "최종낙찰금액_num", "낙찰율_num"
                if x_col in df.columns and y_col in df.columns:
                    plt.figure(figsize=(12, 5))
                    x_vmax = np.percentile(df[x_col].dropna(), 99.5)
                    plt.subplot(1, 2, 1)
                    sns.scatterplot(x=df[x_col].clip(upper=x_vmax), y=df[y_col].dropna(), color="blue", alpha=0.6)
                    plt.title("원본 데이터")
                    plt.subplot(1, 2, 2)
                    sns.scatterplot(x=synth_dp[x_col].clip(upper=x_vmax), y=synth_dp[y_col].dropna(), color="orange", alpha=0.6)
                    plt.title(f"재현 데이터 ({MODEL_METHOD})")
                    plt.tight_layout()
                    
                    # [수정] 파일명에 모델명 포함
                    plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{x_col}_vs_{y_col}_{MODEL_METHOD}.png"), bbox_inches='tight', dpi=150)
                    plt.close()
                
                # 메타데이터 저장 (모델명 포함)
                metadata_summary = {
                    "model_method": MODEL_METHOD, "input_file": INPUT_CSV, 
                    "rows_original": len(df), "rows_synth": len(synth_dp), 
                    "k_target": K_TARGET, "l_target": L_TARGET, "epsilon_dp": EPSILON_DP
                }
                # [수정] 파일명에 모델명 포함
                with open(os.path.join(OUTPUT_DIR, f"synthesis_metadata_{MODEL_METHOD}.json"), "w", encoding="utf-8") as f:
                    json.dump(metadata_summary, f, ensure_ascii=False, indent=2)

                logging.info(f"[Completed] {MODEL_METHOD} 모델을 사용한 프로세스 완료.")
                
        except Exception as e:
            logging.error(f"[Error] 합성 데이터 생성 실패: {e}")

if __name__ == "__main__":
    main()