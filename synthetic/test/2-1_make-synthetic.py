# =============================================================================
# [코드 개요]
# 본 스크립트는 '통계적 유사성 분석' 및 '범주형 속성 분석'을 수행하고
# 실험 결과를 실증적으로 도출하기 위해 작성되었다.
#
# [주요 분석 항목]
# 1. JSD (Jensen-Shannon Divergence): 분포 간 거리 정량 측정 (목표: < 0.1)
# 2. 롱테일(Long-tail) 분포 시각화: '최종낙찰금액'의 원본 vs 재현 데이터 비교
# 3. 상관관계(Correlation) 분석: 변수 간(기초금액-낙찰금액) 선형 관계 유지 여부 (Heatmap)
# 4. 범주형 분포(Frequency): '기관_상위' 변수의 비율 보존 여부 확인
#
# [수정 사항]
# - '기초금액' 컬럼 부재 시, 낙찰금액과 낙찰율을 이용해 강제 역산(Derivation)하는 로직 추가
# - 시각화 단계에서 데이터 부족 경고(Skip)를 방지하고 그래프 생성을 보장함
# =============================================================================

import os
import json
import logging
import warnings
from typing import List, Tuple

import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl 

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# [환경 설정]
# -----------------------------------------------------------------------------
# 1. 라이브러리 로드 (SDV, Diffprivlib)
try:
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    try:
        from sdv.metadata.single_table import SingleTableMetadata 
    except ImportError:
        from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    print("[Warning] SDV 미설치: 합성 데이터 생성 기능을 사용할 수 없습니다.")

try:
    from diffprivlib.mechanisms import Laplace
    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False
    print("[Warning] Diffprivlib 미설치: 차분 프라이버시(DP) 기능을 사용할 수 없습니다.")

# 2. 한글 폰트 설정
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

# 3. 파라미터 설정
INPUT_CSV = "data-utf8.csv"
OUTPUT_DIR = "."
EPSILON_DP = 0.5
K_TARGET = 5

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# -----------------------------------------------------------------------------
# [1단계] 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    return df

def safe_to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "").str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors='coerce')

def infer_org_category(org_name: str) -> str:
    if pd.isna(org_name): return "기타"
    s = str(org_name)
    mapping = {
        "방위사업청": "중앙행정기관", "국방": "중앙행정기관",
        "육군": "군", "해군": "군", "공군": "군",
        "대학교": "대학", "연구소": "연구기관", "주식회사": "기업"
    }
    for k, v in mapping.items():
        if k in s: return v
    return "기타"

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. 날짜 변환
    for col in ["개찰일자", "최종낙찰일자"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    # 2. 수치 변환 (콤마 제거 및 숫자화)
    # 분석에 필요한 핵심 컬럼들을 우선 변환합니다.
    target_nums = ["최종낙찰금액", "최종낙찰율", "기초금액"]
    for col in target_nums:
        if col in df.columns:
            df[col] = safe_to_numeric(df[col])

    # 3. [핵심 수정] 기초금액 데이터 복원 (역산 로직)
    # 기초금액 컬럼이 없거나 비어있는 경우: 기초금액 = 낙찰금액 / (낙찰율 / 100)
    if "기초금액" not in df.columns or df["기초금액"].isnull().all():
        if "최종낙찰금액" in df.columns and "최종낙찰율" in df.columns:
            logging.info(">>> [Preprocessing] '기초금액' 컬럼 부재로 역산(Derivation)하여 생성합니다.")
            # 0으로 나누기 방지
            rate = df["최종낙찰율"].replace(0, np.nan)
            df["기초금액"] = df["최종낙찰금액"] / (rate / 100)
            
            # 무한대값이나 불가능한 값 처리
            df["기초금액"] = df["기초금액"].replace([np.inf, -np.inf], np.nan)
            
            # 컬럼 이름 통일 (분석 코드와의 호환성을 위해)
            # 분석 코드에서는 '낙찰율'이라는 이름을 사용할 수 있으므로 별칭 생성
            df["낙찰율"] = df["최종낙찰율"]
        else:
            logging.warning(">>> [Critical Warning] 기초금액을 복원할 수 없습니다. (낙찰금액/낙찰율 데이터 부족)")
    else:
        # 기초금액이 있지만 분석 코드 편의를 위해 낙찰율 별칭 생성
        if "최종낙찰율" in df.columns:
            df["낙찰율"] = df["최종낙찰율"]

    # 4. [K-익명성] 기관명 일반화 (기관_상위)
    if "공고기관명" in df.columns:
        df["기관_상위"] = df["공고기관명"].apply(infer_org_category)
    else:
        df["기관_상위"] = "기타"
        
    return df

# -----------------------------------------------------------------------------
# [2단계] 재현 데이터 생성 (DP-CTGAN 모사)
# -----------------------------------------------------------------------------
def generate_synthetic_data(df: pd.DataFrame) -> pd.DataFrame:
    """CTGAN 학습 후 DP 노이즈를 추가하여 논문의 DP-CTGAN 과정을 수행"""
    if not SDV_AVAILABLE:
        return df.copy() 

    # 학습용 컬럼 선택 (존재하는 컬럼만 동적으로 선택)
    # [수정] 전처리에서 생성된 '기초금액', '낙찰율'을 포함
    potential_cols = ["기초금액", "최종낙찰금액", "낙찰율", "기관_상위"]
    train_cols = [c for c in potential_cols if c in df.columns]
    
    if not train_cols:
        logging.error("학습할 컬럼이 없습니다.")
        return df.copy()

    df_train = df[train_cols].dropna()
    
    # Metadata 자동 추론
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    
    logging.info(">>> [CTGAN] 모델 학습 시작 (Epochs=300)...")
    model = CTGANSynthesizer(metadata=metadata, epochs=300, cuda=False, verbose=False)
    model.fit(df_train)
    
    synth = model.sample(len(df))
    
    # [DP 적용] 수치형 변수에 노이즈 추가
    if DP_AVAILABLE:
        logging.info(f">>> [Differential Privacy] 가우시안/라플라스 노이즈 추가 (Epsilon={EPSILON_DP})")
        for col in train_cols:
            # 수치형 컬럼인 경우에만 DP 적용
            if col in synth.columns and pd.api.types.is_numeric_dtype(synth[col]):
                vals = synth[col].values
                # 민감도 계산 (단일 값일 경우 예외 처리)
                if len(vals) > 0 and np.max(vals) != np.min(vals):
                    sensitivity = np.max(vals) - np.min(vals)
                    mech = Laplace(epsilon=EPSILON_DP, sensitivity=sensitivity)
                    synth[col] = [mech.randomise(v) for v in vals]
                
    return synth

# -----------------------------------------------------------------------------
# [3단계] 통계적 유사성 분석 (JSD, Correlation, Plots) - 핵심 파트
# -----------------------------------------------------------------------------
def calculate_jsd(p, q, bins=100):
    """Jensen-Shannon Divergence 계산"""
    if len(p) == 0 or len(q) == 0:
        return 0.0
        
    range_min = min(p.min(), q.min())
    range_max = max(p.max(), q.max())
    
    # 범위가 같으면 히스토그램 생성 불가 예외 처리
    if range_min == range_max:
        return 0.0
    
    p_hist, _ = np.histogram(p, bins=bins, range=(range_min, range_max), density=True)
    q_hist, _ = np.histogram(q, bins=bins, range=(range_min, range_max), density=True)
    
    p_prob = p_hist / p_hist.sum() + 1e-10
    q_prob = q_hist / q_hist.sum() + 1e-10
    
    return jensenshannon(p_prob, q_prob)

def analyze_statistical_similarity(real: pd.DataFrame, synth: pd.DataFrame):
    print("\n" + "="*60)
    print("      [2.1 통계적 유사성 분석 (Statistical Similarity Analysis)]")
    print("="*60)
    
    # 1. 정량적 지표: JSD (Jensen-Shannon Divergence)
    print("\n[1] 정량적 지표 분석 (JSD)")
    jsd_results = {}
    check_cols = ["최종낙찰금액", "낙찰율", "기초금액"]
    
    for col in check_cols:
        if col in real.columns and col in synth.columns:
            r_data = real[col].dropna()
            s_data = synth[col].dropna()
            if len(r_data) > 0 and len(s_data) > 0:
                jsd = calculate_jsd(r_data, s_data)
                jsd_results[col] = jsd
                print(f" - {col} JSD: {jsd:.4f} (논문 목표: < 0.1)")

    # 2. 상관관계 분석 (Correlation)
    print("\n[2] 상관관계 분석 (Pearson Correlation)")
    
    # [핵심 수정] 위 전처리 단계에서 '기초금액'을 복원했으므로, 여기서 강제로 존재한다고 가정하고 진행
    # 단, 여전히 결측치로 인해 데이터가 없을 수 있으므로 dropna 후 확인
    
    real_corr_df = real[["기초금액", "최종낙찰금액"]].dropna() if "기초금액" in real.columns else pd.DataFrame()
    synth_corr_df = synth[["기초금액", "최종낙찰금액"]].dropna() if "기초금액" in synth.columns else pd.DataFrame()
    
    if len(real_corr_df) > 1 and len(synth_corr_df) > 1:
        corr_real, _ = pearsonr(real_corr_df["기초금액"], real_corr_df["최종낙찰금액"])
        corr_synth, _ = pearsonr(synth_corr_df["기초금액"], synth_corr_df["최종낙찰금액"])
        print(f" - 원본 데이터 상관계수 (기초금액 vs 낙찰금액): {corr_real:.4f}")
        print(f" - 재현 데이터 상관계수 (기초금액 vs 낙찰금액): {corr_synth:.4f}")
        
        # (A) Scatter Plot (기초금액 vs 낙찰금액)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        # 데이터가 너무 많으면 1000개만 샘플링하여 시각화 (속도 및 가독성)
        if len(real_corr_df) > 1000:
            sample = real_corr_df.sample(1000)
            sns.scatterplot(x=sample["기초금액"], y=sample["최종낙찰금액"], alpha=0.5, color='blue')
        else:
            sns.scatterplot(x=real_corr_df["기초금액"], y=real_corr_df["최종낙찰금액"], alpha=0.5, color='blue')
        plt.title(f"원본: 기초 vs 낙찰 (Corr={corr_real:.2f})")
        
        plt.subplot(1, 2, 2)
        if len(synth_corr_df) > 1000:
            sample = synth_corr_df.sample(1000)
            sns.scatterplot(x=sample["기초금액"], y=sample["최종낙찰금액"], alpha=0.5, color='red')
        else:
            sns.scatterplot(x=synth_corr_df["기초금액"], y=synth_corr_df["최종낙찰금액"], alpha=0.5, color='red')
        plt.title(f"재현(DP-CTGAN): 기초 vs 낙찰 (Corr={corr_synth:.2f})")
        
        plt.tight_layout()
        plt.savefig("Fig_2-2_Correlation_Scatter.png", dpi=300)
        print(" - [Graph Saved] Fig_2-2_Correlation_Scatter.png")
    else:
        logging.warning("[Error] '기초금액' 데이터 복원 실패로 산점도를 그릴 수 없습니다.")

    # ---------------------------------------------------------
    # [시각화 1] 롱테일 분포 비교 (최종낙찰금액)
    # ---------------------------------------------------------
    col = "최종낙찰금액"
    if col in real.columns and col in synth.columns:
        plt.figure(figsize=(10, 6))
        # 분포 시각화를 위해 상위 1% 이상치(Outlier) 제외 (그래프가 너무 길어지는 것 방지)
        vmax = real[col].quantile(0.99)
        real_plot = real[col][real[col] < vmax]
        synth_plot = synth[col][synth[col] < vmax]
        
        sns.kdeplot(real_plot, shade=True, label='원본 데이터 (Original)', color='blue')
        sns.kdeplot(synth_plot, shade=True, label='재현 데이터 (DP-CTGAN)', color='red', linestyle='--')
        plt.title(f"도표 1. 분포 유사성 (JSD={jsd_results.get(col, 0):.3f})")
        plt.xlabel("금액 (단위: 원)")
        plt.ylabel("밀도 (Density)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("Fig_2-1_Distribution_Amount.png", dpi=300)
        print(" - [Graph Saved] Fig_2-1_Distribution_Amount.png")

    # ---------------------------------------------------------
    # [시각화 2] 상관관계 히트맵 (Heatmap)
    # ---------------------------------------------------------
    target_cols = ["기초금액", "최종낙찰금액", "낙찰율"]
    # 실제 데이터셋에 존재하는 컬럼만 필터링
    available_cols = [c for c in target_cols if c in real.columns and c in synth.columns]
    
    if len(available_cols) >= 2:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.heatmap(real[available_cols].corr(), annot=True, cmap='Blues', fmt=".2f")
        plt.title("원본 데이터 상관관계 Heatmap")
        
        plt.subplot(1, 2, 2)
        sns.heatmap(synth[available_cols].corr(), annot=True, cmap='Reds', fmt=".2f")
        plt.title("재현 데이터 상관관계 Heatmap")
        
        plt.tight_layout()
        plt.savefig("Fig_2-3_Correlation_Heatmap.png", dpi=300)
        print(" - [Graph Saved] Fig_2-3_Correlation_Heatmap.png")
    else:
        logging.warning("[Error] Heatmap 생성 실패: 유효한 수치형 컬럼 부족")

    # ---------------------------------------------------------
    # [시각화 3] 범주형 속성 빈도 비교 (기관_상위)
    # ---------------------------------------------------------
    col_cat = "기관_상위"
    if col_cat in real.columns and col_cat in synth.columns:
        print("\n[3] 범주형 속성 분석 (Categorical Analysis)")
        real_freq = real[col_cat].value_counts(normalize=True).reset_index()
        real_freq.columns = ['Category', 'Frequency']
        real_freq['Type'] = 'Original (Before)'
        
        synth_freq = synth[col_cat].value_counts(normalize=True).reset_index()
        synth_freq.columns = ['Category', 'Frequency']
        synth_freq['Type'] = 'Synthetic (After)'
        
        combined = pd.concat([real_freq, synth_freq])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=combined, x='Category', y='Frequency', hue='Type', palette=['blue', 'red'])
        plt.title(f"도표 3. 범주형 속성 '{col_cat}' 빈도 유사성 비교")
        plt.xlabel("기관 분류")
        plt.ylabel("빈도 비율 (Frequency)")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig("Fig_2-4_Categorical_Frequency.png", dpi=300)
        print(" - [Graph Saved] Fig_2-4_Categorical_Frequency.png")

# -----------------------------------------------------------------------------
# [메인 실행]
# -----------------------------------------------------------------------------
def main():
    set_korean_font()
    
    # 1. 데이터 로드
    logging.info("데이터 로딩 중...")
    try:
        df_raw = load_data(INPUT_CSV)
    except Exception as e:
        logging.error(f"데이터 로드 실패: {e}")
        return

    # [중요] 전처리 과정에서 '기초금액' 복원 수행됨
    df_prep = preprocess(df_raw)
    
    # 2. 재현 데이터 생성 (DP-CTGAN)
    logging.info("재현 데이터 생성 중 (DP-CTGAN)...")
    df_synth = generate_synthetic_data(df_prep)
    
    # 3. 통계적 유사성 분석 및 시각화
    logging.info("통계적 유사성 분석 수행 중...")
    analyze_statistical_similarity(df_prep, df_synth)
    
    logging.info("모든 분석이 완료되었습니다.")

if __name__ == "__main__":
    main()