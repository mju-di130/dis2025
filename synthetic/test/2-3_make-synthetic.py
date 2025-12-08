# =============================================================================
# [실험: 원본 데이터 기반 재현 데이터 생성 및 검증 (최종본)]
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings
import os

# 라이브러리 임포트 (SDV, Scipy 등)
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr
except ImportError:
    print("[Error] 필수 라이브러리가 없습니다. (!pip install sdv scipy seaborn)")

warnings.filterwarnings("ignore")

# 1. 한글 폰트 설정 (깨짐 방지)
def set_korean_font():
    font_paths = ['C:/Windows/Fonts/malgun.ttf', '/System/Library/Fonts/Supplemental/AppleGothic.ttf', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
    for path in font_paths:
        try:
            font_name = fm.FontProperties(fname=path).get_name()
            plt.rc('font', family=font_name)
            plt.rc('axes', unicode_minus=False)
            print(f"[System] 폰트 설정 완료: {font_name}")
            return
        except:
            continue
    print("[Warning] 한글 폰트를 찾지 못했습니다.")

# -----------------------------------------------------------------------------
# [Step 1] 데이터 전처리 및 파생변수 생성 (Feature Engineering)
# -----------------------------------------------------------------------------
def preprocess_data(file_path):
    # 데이터 로드
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"[System] 원본 데이터 로드: {len(df)}건")

    # 1. 수치형 변환 (콤마 제거)
    def clean_currency(x):
        try:
            return float(str(x).replace(',', ''))
        except:
            return np.nan

    df['최종낙찰금액_num'] = df['최종낙찰금액'].apply(clean_currency)
    df['낙찰율_num'] = df['최종낙찰율'].apply(clean_currency)

    # 2. [핵심] 기초금액 역산 (Derivation)
    df = df[df['낙찰율_num'] > 0].copy()
    df['기초금액_num'] = df['최종낙찰금액_num'] / (df['낙찰율_num'] / 100)
    
    # 이상치(Outlier) 제거 (상위 1% 제거하여 분포 안정화)
    q99 = df['최종낙찰금액_num'].quantile(0.99)
    df = df[df['최종낙찰금액_num'] <= q99].copy()

    # 3. 기관명 범주화 (K-익명성 일반화)
    def category_agency(name):
        name = str(name)
        if '방위사업청' in name or '국방' in name: return '중앙행정기관'
        if any(x in name for x in ['육군', '해군', '공군', '부대']): return '군'
        if any(x in name for x in ['연구소', '연구원', '과학']): return '연구기관'
        return '기타'

    df['기관_상위'] = df['공고기관명'].apply(category_agency)

    # 4. 로그 변환 (학습 성능 향상용)
    df['기초금액_log'] = np.log1p(df['기초금액_num'])
    df['낙찰금액_log'] = np.log1p(df['최종낙찰금액_num'])

    # 학습에 사용할 컬럼만 선택
    target_cols = ['기초금액_log', '낙찰금액_log', '낙찰율_num', '기관_상위']
    return df[target_cols].dropna(), df  # 학습용 데이터, 원본 전체 데이터 반환

# -----------------------------------------------------------------------------
# [Step 2] 재현 데이터 생성 (DP-CTGAN Simulation)
# -----------------------------------------------------------------------------
def generate_synthetic(df_train, original_len):
    print("[System] 재현 데이터 생성 시작 (CTGAN)...")
    
    # 메타데이터 자동 감지
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)

    # 모델 학습 (속도를 위해 Epochs=200 설정, 실제 논문용은 500 권장)
    model = CTGANSynthesizer(metadata, epochs=500, verbose=True)
    model.fit(df_train)

    # 데이터 생성
    synth = model.sample(original_len)

    # 로그 역변환 (복원) -> 원래 금액 단위로
    synth['기초금액_num'] = np.expm1(synth['기초금액_log'])
    synth['최종낙찰금액_num'] = np.expm1(synth['낙찰금액_log'])
    
    # 범위 보정 (음수 제거)
    synth['기초금액_num'] = synth['기초금액_num'].clip(lower=0)
    synth['최종낙찰금액_num'] = synth['최종낙찰금액_num'].clip(lower=0)

    # 학습용 로그 컬럼 제거 (최종 데이터셋 정리)
    synth_final = synth[['기초금액_num', '최종낙찰금액_num', '낙찰율_num', '기관_상위']]
    
    return synth_final

# -----------------------------------------------------------------------------
# [Step 3] 시각화 및 검증 (Visualization)
# -----------------------------------------------------------------------------
def evaluate_results(real_df, synth_df):
    set_korean_font()
    print("[System] 실험 결과 시각화 생성 중...")

    # JSD 계산 함수
    def get_jsd(p, q):
        p_hist, _ = np.histogram(p, bins=100, density=True)
        q_hist, _ = np.histogram(q, bins=100, density=True)
        return jensenshannon(p_hist+1e-10, q_hist+1e-10)

    jsd_val = get_jsd(real_df['최종낙찰금액_num'], synth_df['최종낙찰금액_num'])
    print(f" >> 최종낙찰금액 JSD: {jsd_val:.4f} (목표: < 0.1)")

    # 1. 롱테일 분포 비교 (KDE Plot)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(real_df['최종낙찰금액_num'], fill=True, label='원본 (Original)', color='blue')
    sns.kdeplot(synth_df['최종낙찰금액_num'], fill=True, label='재현 (Synthetic)', color='red', linestyle='--')
    plt.title(f"도표 1. 최종낙찰금액 분포 유사성 (JSD={jsd_val:.3f})")
    plt.xlabel("금액 (원)"); plt.legend()
    plt.savefig("Exp_Result_1_Distribution.png")
    
    # 2. 산점도 (Scatter Plot)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=real_df['기초금액_num'], y=real_df['최종낙찰금액_num'], alpha=0.3, color='blue')
    plt.title("원본: 기초금액 vs 낙찰금액")
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=synth_df['기초금액_num'], y=synth_df['최종낙찰금액_num'], alpha=0.3, color='red')
    plt.title("재현: 기초금액 vs 낙찰금액")
    plt.tight_layout()
    plt.savefig("Exp_Result_2_Scatter.png")

    # 3. 히트맵 (Correlation)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(real_df[['기초금액_num', '최종낙찰금액_num', '낙찰율_num']].corr(), annot=True, cmap='Blues', fmt=".2f")
    plt.title("원본 상관관계")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(synth_df[['기초금액_num', '최종낙찰금액_num', '낙찰율_num']].corr(), annot=True, cmap='Reds', fmt=".2f")
    plt.title("재현 상관관계")
    plt.tight_layout()
    plt.savefig("Exp_Result_3_Heatmap.png")

    # 4. 범주형 빈도 (Bar Chart)
    real_cnt = real_df['기관_상위'].value_counts(normalize=True).reset_index()
    real_cnt['Type'] = 'Original'
    synth_cnt = synth_df['기관_상위'].value_counts(normalize=True).reset_index()
    synth_cnt['Type'] = 'Synthetic'
    
    combined = pd.concat([real_cnt, synth_cnt])
    combined.columns = ['Category', 'Frequency', 'Type']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=combined, x='Category', y='Frequency', hue='Type', palette=['blue', 'red'])
    plt.title("도표 4. 기관 분류별 빈도 유사성")
    plt.savefig("Exp_Result_4_Category.png")

    print("[System] 모든 결과 이미지가 저장되었습니다.")

# -----------------------------------------------------------------------------
# [Main Execution]
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    file_name = "data.csv"
    
    # 1. 전처리
    train_data, raw_data = preprocess_data(file_name)
    
    # 2. 재현 데이터 생성
    synthetic_data = generate_synthetic(train_data, len(raw_data))
    
    # 3. [추가] 생성된 재현 데이터 CSV 파일로 저장
    output_csv_name = "synthetic_data.csv"
    synthetic_data.to_csv(output_csv_name, index=False, encoding='utf-8-sig')
    print(f"[System] 재현 데이터가 '{output_csv_name}' 파일로 저장되었습니다.")
    
    # 4. 평가 및 시각화 (원본 데이터에는 전처리된 컬럼을 붙여서 비교)
    # 원본 비교용 데이터셋 구성
    raw_comp = raw_data[['최종낙찰금액_num', '낙찰율_num', '기초금액_num', '기관_상위']].dropna()
    
    evaluate_results(raw_comp, synthetic_data)