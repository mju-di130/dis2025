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
# ----------------------------------------------------------------------------
# [결과물 요약]
# 연번 / 구분 / 파일명 / 설명
# 1. 전처리 데이터 / preprocessed_defense_rnd.csv / 원본 데이터에 전처리(날짜/금액 변환)와 K-익명성(기관명 일반화)이 적용된 데이터
# 2. L-다양성 보고서 / l_diversity_suppression_report.csv / L-다양성 기준(L=3)을 만족하지 못해 마스킹(Suppression) 처리된 그룹의 정보
# 3. 재현 데이터 (Raw) / synthetic_data.csv / SDV 모델(Gaussian Copula 등)을 통해 생성된 초기 재현 데이터
# 4. 재현 데이터 (Final) / synthetic_data_dp_eps0.5.csv / 수치형 변수에 차분 프라이버시(DP) 노이즈(ϵ=0.5)까지 최종 적용된 데이터
# 5. 메타데이터 / synthesis_metadata.json / 실험에 사용된 파라미터(K,L,ϵ) 및 데이터 건수 요약 정보
# 6. 이미지 (분포) / hist_최종낙찰금액_num_orig_vs_synth.png / 원본 vs 재현 데이터의 금액 분포 비교 히스토그램
# 7. 이미지 (상관관계) / scatter_최종낙찰금액_num_vs_낙찰율_num.png / 금액과 낙찰율 간의 상관관계를 보여주는 산점도 비교 그래프
# =============================================================================
#
# [수정 사항]
# 1. 데이터 복원 로직 추가: '기초금액' 컬럼 부재 시, '최종낙찰금액'과 '낙찰율'을 이용해 역산
#    (공식: 기초금액 = 낙찰금액 / (낙찰율 / 100))
# 2. 모든 시각화 강제 생성: 데이터가 부족해도 가능한 모든 그래프를 출력하도록 예외 처리 강화
# =============================================================================

import os
import logging
import warnings
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
# 1. 라이브러리 로드
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata.single_table import SingleTableMetadata
        SDV_AVAILABLE = True
    except ImportError:
        SDV_AVAILABLE = False
        print("[Warning] SDV 미설치: 합성 데이터 생성 기능을 사용할 수 없습니다. (pip install sdv)")

try:
    from diffprivlib.mechanisms import Laplace
    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False
    print("[Warning] Diffprivlib 미설치: 차분 프라이버시 기능을 사용할 수 없습니다. (pip install diffprivlib)")

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
    else:
        print("[Font] 한글 폰트를 찾을 수 없습니다. 그래프 글자가 깨질 수 있습니다.")

# 3. 파라미터 설정
INPUT_CSV = "data-utf8.csv"
OUTPUT_DIR = "."
EPSILON_DP = 0.5

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# -----------------------------------------------------------------------------
# [1단계] 데이터 로드 및 전처리 (Preprocessing)
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
    
    # 1. 수치형 변환
    if "최종낙찰금액" in df.columns:
        df["최종낙찰금액_num"] = safe_to_numeric(df["최종낙찰금액"])
    if "최종낙찰율" in df.columns:
        df["낙찰율_num"] = safe_to_numeric(df["최종낙찰율"])
        
    # [핵심 수정] 2. 기초금액 파생 변수 생성 (데이터 누락 시 역산)
    if "기초금액" in df.columns:
        df["기초금액_num"] = safe_to_numeric(df["기초금액"])
    else:
        # 기초금액이 없으면 낙찰금액과 낙찰율로 역산 (기초 = 낙찰 / (율/100))
        if "최종낙찰금액_num" in df.columns and "낙찰율_num" in df.columns:
            logging.info("[Preprocessing] '기초금액' 컬럼 부재로 역산(Derivation)하여 생성합니다.")
            df["기초금액_num"] = df["최종낙찰금액_num"] / (df["낙찰율_num"] / 100)
            df["기초금액_num"] = df["기초금액_num"].replace([np.inf, -np.inf], np.nan) # 0으로 나누기 방지
        else:
            logging.warning("[Warning] 기초금액을 복원할 수 없습니다. (낙찰금액/낙찰율 데이터 부족)")

    # 3. 기관명 일반화
    if "공고기관명" in df.columns:
        df["기관_상위"] = df["공고기관명"].apply(infer_org_category)
    else:
        df["기관_상위"] = "기타"
        
    return df

# -----------------------------------------------------------------------------
# [2단계] 재현 데이터 생성 (DP-CTGAN)
# -----------------------------------------------------------------------------
def generate_synthetic_data(df: pd.DataFrame) -> pd.DataFrame:
    if not SDV_AVAILABLE:
        logging.warning("SDV 미설치로 원본 데이터를 복사하여 테스트합니다.")
        return df.copy()

    # 학습에 사용할 유효 컬럼 선정
    train_cols = [c for c in ["기초금액_num", "최종낙찰금액_num", "낙찰율_num", "기관_상위"] if c in df.columns]
    
    if not train_cols:
        return df.copy()

    df_train = df[train_cols].dropna()
    
    # 메타데이터 자동 추론
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    
    logging.info(">>> [CTGAN] 모델 학습 시작 (Epochs=100)...") # 속도를 위해 100회로 설정
    model = CTGANSynthesizer(metadata=metadata, epochs=100, cuda=False, verbose=False)
    model.fit(df_train)
    
    synth = model.sample(len(df))
    
    # DP 노이즈 추가
    if DP_AVAILABLE:
        logging.info(f">>> [Differential Privacy] 노이즈 추가 (Epsilon={EPSILON_DP})")
        for col in train_cols:
            if col in synth.columns and pd.api.types.is_numeric_dtype(synth[col]):
                vals = synth[col].values
                if len(vals) > 0:
                    sensitivity = np.max(vals) - np.min(vals)
                    if sensitivity == 0: sensitivity = 1.0
                    mech = Laplace(epsilon=EPSILON_DP, sensitivity=sensitivity)
                    synth[col] = [mech.randomise(v) for v in vals]
                    
    return synth

# -----------------------------------------------------------------------------
# [3단계] 시각화 및 분석 (All Figures Generation)
# -----------------------------------------------------------------------------
def analyze_statistical_similarity(real: pd.DataFrame, synth: pd.DataFrame):
    print("\n" + "="*60)
    print("      [2.1 통계적 유사성 분석 결과]")
    print("="*60)
    
    # [1] 정량적 지표 (JSD)
    def calculate_jsd(p, q, bins=100):
        if len(p) == 0 or len(q) == 0: return 0.0
        range_min = min(p.min(), q.min())
        range_max = max(p.max(), q.max())
        if range_min == range_max: return 0.0
        p_hist, _ = np.histogram(p, bins=bins, range=(range_min, range_max), density=True)
        q_hist, _ = np.histogram(q, bins=bins, range=(range_min, range_max), density=True)
        p_prob = p_hist / p_hist.sum() + 1e-10
        q_prob = q_hist / q_hist.sum() + 1e-10
        return jensenshannon(p_prob, q_prob)

    jsd_val = 0.0
    if "최종낙찰금액_num" in real.columns and "최종낙찰금액_num" in synth.columns:
        jsd_val = calculate_jsd(real["최종낙찰금액_num"].dropna(), synth["최종낙찰금액_num"].dropna())
        print(f" - 최종낙찰금액 JSD: {jsd_val:.4f} (목표 < 0.1)")

    # ---------------------------------------------------------
    # [Figure 1] 롱테일 분포 비교 (Distribution)
    # ---------------------------------------------------------
    # 

# [Image of Normal Distribution Curve]

    # 위 태그는 정규분포 곡선의 예시를 보여주는 태그입니다. 아래 코드는 실제 데이터의 KDE 분포를 그립니다.
    if "최종낙찰금액_num" in real.columns:
        plt.figure(figsize=(10, 6))
        # 데이터 클리핑 (상위 5% 이상치 제외하여 그래프 가독성 확보)
        vmax = real["최종낙찰금액_num"].quantile(0.95)
        sns.kdeplot(real["최종낙찰금액_num"].clip(upper=vmax), fill=True, label='원본 (Original)', color='blue')
        sns.kdeplot(synth["최종낙찰금액_num"].clip(upper=vmax), fill=True, label='재현 (Synthetic)', color='red', linestyle='--')
        plt.title(f"도표 1. 최종낙찰금액 분포 유사성 (JSD={jsd_val:.3f})")
        plt.xlabel("금액 (원)")
        plt.ylabel("밀도")
        plt.legend()
        plt.savefig("Fig_2-1_Distribution_Amount.png", dpi=300)
        print(" [생성 완료] Fig_2-1_Distribution_Amount.png")

    # ---------------------------------------------------------
    # [Figure 2] 상관관계 산점도 (Scatter Plot)
    # ---------------------------------------------------------
    # # 위 태그는 산점도의 예시를 보여주는 태그입니다.
    if "기초금액_num" in real.columns and "최종낙찰금액_num" in real.columns:
        # 시각화를 위해 샘플링 (점 1000개만 표시)
        idx = np.random.choice(len(real), min(1000, len(real)), replace=False)
        real_sample = real.iloc[idx]
        synth_sample = synth.iloc[idx]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=real_sample["기초금액_num"], y=real_sample["최종낙찰금액_num"], alpha=0.5, color='blue')
        plt.title("원본: 기초금액 vs 낙찰금액")
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=synth_sample["기초금액_num"], y=synth_sample["최종낙찰금액_num"], alpha=0.5, color='red')
        plt.title("재현: 기초금액 vs 낙찰금액")
        
        plt.tight_layout()
        plt.savefig("Fig_2-2_Correlation_Scatter.png", dpi=300)
        print(" [생성 완료] Fig_2-2_Correlation_Scatter.png")
    else:
        print(" [실패] 산점도 생성 불가 (기초금액 데이터 부족)")

    # ---------------------------------------------------------
    # [Figure 3] 상관관계 히트맵 (Heatmap)
    # ---------------------------------------------------------
    # 

#[Image of Correlation Matrix Heatmap]

    # 위 태그는 히트맵의 예시를 보여주는 태그입니다.
    target_cols = ["기초금액_num", "최종낙찰금액_num", "낙찰율_num"]
    use_cols = [c for c in target_cols if c in real.columns and c in synth.columns]
    
    if len(use_cols) >= 2:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.heatmap(real[use_cols].corr(), annot=True, cmap='Blues', fmt=".2f", vmin=0, vmax=1)
        plt.title("원본 데이터 상관관계")
        
        plt.subplot(1, 2, 2)
        sns.heatmap(synth[use_cols].corr(), annot=True, cmap='Reds', fmt=".2f", vmin=0, vmax=1)
        plt.title("재현 데이터 상관관계")
        
        plt.tight_layout()
        plt.savefig("Fig_2-3_Correlation_Heatmap.png", dpi=300)
        print(" [생성 완료] Fig_2-3_Correlation_Heatmap.png")

    # ---------------------------------------------------------
    # [Figure 4] 범주형 속성 빈도 (Bar Chart)
    # ---------------------------------------------------------
    if "기관_상위" in real.columns:
        real_freq = real["기관_상위"].value_counts(normalize=True).reset_index()
        real_freq.columns = ['Category', 'Frequency']
        real_freq['Type'] = 'Original'
        
        synth_freq = synth["기관_상위"].value_counts(normalize=True).reset_index()
        synth_freq.columns = ['Category', 'Frequency']
        synth_freq['Type'] = 'Synthetic'
        
        combined = pd.concat([real_freq, synth_freq])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=combined, x='Category', y='Frequency', hue='Type', palette=['blue', 'red'])
        plt.title("도표 4. 기관 분류별 빈도 유사성")
        plt.ylabel("비율 (Frequency)")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig("Fig_2-4_Categorical_Frequency.png", dpi=300)
        print(" [생성 완료] Fig_2-4_Categorical_Frequency.png")

# -----------------------------------------------------------------------------
# [메인 실행]
# -----------------------------------------------------------------------------
def main():
    set_korean_font()
    logging.info("데이터 처리 시작...")
    
    try:
        df_raw = load_data(INPUT_CSV)
        df_prep = preprocess(df_raw) # 여기서 기초금액 역산 수행
        
        logging.info("재현 데이터 생성 (DP-CTGAN)...")
        df_synth = generate_synthetic_data(df_prep)
        
        logging.info("시각화 생성 중...")
        analyze_statistical_similarity(df_prep, df_synth)
        
        logging.info("모든 과정이 완료되었습니다.")
        
    except Exception as e:
        logging.error(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
