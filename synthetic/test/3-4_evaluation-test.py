# =============================================================================
# [실험 코드: 무결성 수정 버전]
# 원본 vs 재현 데이터 구조 일치화 및 성능/보안 검증
#
# [수정 사항]
# 1. 파일명 수정: data.csv, synthetic_data.csv 로딩
# 2. 구조 동기화: 원본 데이터(Real)를 전처리하여 재현 데이터(Synth)와 동일한 컬럼/타입으로 변환
# 3. 예외 처리: 결측치 및 무한대 값 제거로 머신러닝 오류 방지
#
# [수정 사항]
# 1. ValueError 해결: '기관_상위' 등 문자열 컬럼을 강제로 Label Encoding 처리
# 2. 타입 변환: 학습 전에 모든 데이터를 수치형(float)으로 변환하여 문자열 잔존 방지
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl 
import warnings

# 머신러닝 및 평가 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import pairwise_distances
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# [환경 설정]
# -----------------------------------------------------------------------------
def set_korean_font():
    font_paths = ['C:/Windows/Fonts/malgun.ttf', '/System/Library/Fonts/Supplemental/AppleGothic.ttf', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
    for path in font_paths:
        try:
            if os.path.exists(path):
                font_name = fm.FontProperties(fname=path).get_name()
                mpl.rc('font', family=font_name)
                mpl.rc('axes', unicode_minus=False)
                return
        except:
            continue

# 파일명 설정 (업로드된 파일명 기준)
REAL_DATA_PATH = "data.csv"
SYNTH_DATA_PATH = "synthetic_data_full_columns.csv"

# -----------------------------------------------------------------------------
# [1단계] 데이터 로드 및 전처리 (Preprocessing)
# -----------------------------------------------------------------------------
def clean_currency(x):
    """문자열 금액('1,000')을 실수형(1000.0)으로 변환"""
    try:
        return float(str(x).replace(',', ''))
    except:
        return 0.0 # Nan 대신 0.0으로 처리

def infer_org_category(org_name: str) -> str:
    """기관명 일반화"""
    if pd.isna(org_name): return "기타"
    s = str(org_name)
    mapping = {"방위사업청": "중앙행정기관", "국방": "중앙행정기관", "육군": "군", "해군": "군", "공군": "군", "대학교": "대학", "연구소": "연구기관", "주식회사": "기업"}
    for k, v in mapping.items():
        if k in s: return v
    return "기타"

def load_and_align_data():
    """원본과 재현 데이터의 구조를 머신러닝 학습이 가능하도록 통일"""
    # 1. 파일 로드
    if not os.path.exists(REAL_DATA_PATH) or not os.path.exists(SYNTH_DATA_PATH):
        raise FileNotFoundError("데이터 파일을 찾을 수 없습니다.")
    
    try:
        df_real_raw = pd.read_csv(REAL_DATA_PATH, encoding='utf-8')
    except:
        df_real_raw = pd.read_csv(REAL_DATA_PATH, encoding='cp949')
        
    df_synth = pd.read_csv(SYNTH_DATA_PATH, encoding='utf-8-sig')
    
    print(f"[Data Loaded] Real: {df_real_raw.shape}, Synth: {df_synth.shape}")

    # 2. 원본 데이터(Real)를 재현 데이터(Synth) 형식으로 변환
    df_real = pd.DataFrame()
    
    # (1) 수치형 변환
    if '최종낙찰금액' in df_real_raw.columns:
        df_real['최종낙찰금액_num'] = df_real_raw['최종낙찰금액'].apply(clean_currency)
    if '최종낙찰율' in df_real_raw.columns:
        df_real['낙찰율_num'] = df_real_raw['최종낙찰율'].apply(clean_currency)
    
    # 기초금액 복원
    if '기초금액_num' not in df_real.columns:
        if '기초금액' in df_real_raw.columns:
             df_real['기초금액_num'] = df_real_raw['기초금액'].apply(clean_currency)
        else:
            # 기초금액 역산 (0 나누기 방지)
            df_real['기초금액_num'] = df_real['최종낙찰금액_num'] / (df_real['낙찰율_num'].replace(0, 1) / 100)
            
    # (2) 범주형 변환 (기관_상위) - 여기서 '군', '대학' 등의 문자열이 생성됨
    if '공고기관명' in df_real_raw.columns:
        df_real['기관_상위'] = df_real_raw['공고기관명'].apply(infer_org_category)
    else:
        df_real['기관_상위'] = "기타"

    # 3. 공통 컬럼만 선택
    common_cols = [c for c in df_synth.columns if c in df_real.columns]
    
    df_real = df_real[common_cols].fillna(0)
    df_synth = df_synth[common_cols].fillna(0)
    
    # 4. 타겟 변수 생성
    df_real['Target'] = np.where(df_real['낙찰율_num'] < 88.0, 1, 0)
    df_synth['Target'] = np.where(df_synth['낙찰율_num'] < 88.0, 1, 0)
    
    print(f"[Preprocessing] 공통 컬럼: {common_cols}")
    
    return df_real, df_synth

# -----------------------------------------------------------------------------
# [2단계] 머신러닝 효용성 평가 (TSTR) - ★ 핵심 수정 부분 ★
# -----------------------------------------------------------------------------
def evaluate_ml_efficacy(real, synth):
    print("\n" + "="*50)
    print(" [실험 1] 머신러닝 효용성 평가 (TSTR)")
    print("="*50)
    
    target_col = 'Target'
    
    # [수정] 모든 문자열 컬럼(Object)을 찾아서 숫자로 변환 (Label Encoding)
    # 원본과 재현 데이터를 합쳐서 fit 해야 인코딩 맵핑이 일치함
    combined = pd.concat([real, synth], axis=0)
    
    # 문자열 컬럼 자동 탐지
    object_cols = real.select_dtypes(include=['object', 'category']).columns
    print(f"[Encoding] 인코딩 대상 컬럼: {object_cols.tolist()}")
    
    for col in object_cols:
        le = LabelEncoder()
        # 모든 값을 문자열로 변환 후 학습 (int/str 섞임 방지)
        le.fit(combined[col].astype(str))
        real[col] = le.transform(real[col].astype(str))
        synth[col] = le.transform(synth[col].astype(str))

    # [수정] 학습 데이터는 모두 float 형으로 강제 변환 (에러 원천 차단)
    real = real.astype(float)
    synth = synth.astype(float)

    # Train/Test 분리
    X_real = real.drop(target_col, axis=1)
    y_real = real[target_col]
    X_synth = synth.drop(target_col, axis=1)
    y_synth = synth[target_col]
    
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )
    
    # 1. TRTR (Train on Real, Test on Real)
    model_real = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model_real.fit(X_train_real, y_train_real)
    pred_real = model_real.predict(X_test_real)
    f1_trtr = f1_score(y_test_real, pred_real, average='macro')
    
    # 2. TSTR (Train on Synthetic, Test on Real)
    model_synth = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model_synth.fit(X_synth, y_synth) 
    pred_synth = model_synth.predict(X_test_real) 
    f1_tstr = f1_score(y_test_real, pred_synth, average='macro')
    
    print(f" 1. TRTR (원본 학습) F1-Score: {f1_trtr:.4f}")
    print(f" 2. TSTR (재현 학습) F1-Score: {f1_tstr:.4f}")
    
    if f1_trtr > 0:
        print(f" -> 성능 보존율: {(f1_tstr/f1_trtr)*100:.2f}%")
    else:
        print(" -> 성능 보존율 계산 불가 (TRTR score is 0)")
    
    return f1_trtr, f1_tstr

# -----------------------------------------------------------------------------
# [3단계] 프라이버시 안전성 검증 (DCR)
# -----------------------------------------------------------------------------
def calculate_dcr(real, synth):
    print("\n" + "="*50)
    print(" [실험 2] 프라이버시 안전성 검증 (DCR)")
    print("="*50)
    
    # DCR 계산을 위해 문자열 인코딩 및 정규화
    target_col = 'Target'
    combined = pd.concat([real, synth], axis=0)
    
    # 인코딩 (여기서도 문자열이 있으면 에러나므로 다시 처리)
    object_cols = real.select_dtypes(include=['object', 'category']).columns
    for col in object_cols:
        le = LabelEncoder()
        le.fit(combined[col].astype(str))
        real[col] = le.transform(real[col].astype(str))
        synth[col] = le.transform(synth[col].astype(str))
        
    n_sample = min(2000, len(real), len(synth))
    # 샘플링 시 인덱스 리셋
    real = real.reset_index(drop=True)
    synth = synth.reset_index(drop=True)
    
    real_sample = real.sample(n=n_sample, random_state=42).drop(target_col, axis=1, errors='ignore')
    synth_sample = synth.sample(n=n_sample, random_state=42).drop(target_col, axis=1, errors='ignore')
    
    scaler = MinMaxScaler()
    real_norm = scaler.fit_transform(real_sample)
    synth_norm = scaler.transform(synth_sample)
    
    dists = pairwise_distances(synth_norm, real_norm, metric='euclidean')
    dcr_values = dists.min(axis=1)
    
    mean_dcr = np.mean(dcr_values)
    min_dcr = np.min(dcr_values)
    exact_matches = np.sum(dcr_values == 0)
    
    print(f" 1. 평균 거리 (Mean DCR): {mean_dcr:.4f}")
    print(f" 2. 최소 거리 (Min DCR): {min_dcr:.4f}")
    print(f" 3. 원본 일치 건수 (DCR=0): {exact_matches}건")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(dcr_values, kde=True, color='green', bins=50)
    plt.axvline(x=np.percentile(dcr_values, 5), color='red', linestyle='--', label='Risk Threshold (5%)')
    plt.title("Distance to Closest Record (DCR) Distribution")
    plt.xlabel("Euclidean Distance"); plt.ylabel("Count")
    plt.legend()
    plt.savefig("Fig_DCR_Security_Analysis.png", dpi=150)
    print(" -> [Saved] Fig_DCR_Security_Analysis.png")

# -----------------------------------------------------------------------------
# [Main Execution]
# -----------------------------------------------------------------------------
def main():
    set_korean_font()
    
    # 1. 데이터 로드
    try:
        df_real, df_synth = load_and_align_data()
    except Exception as e:
        print(f"[Error] 데이터 준비 중 오류 발생: {e}")
        return

    # 2. 머신러닝 평가 (데이터 복사본 전달)
    evaluate_ml_efficacy(df_real.copy(), df_synth.copy())
    
    # 3. 보안성 평가
    calculate_dcr(df_real.copy(), df_synth.copy())
    
    # 4. 결과 파일 저장
    with open("evaluation_summary.txt", "w", encoding='utf-8') as f:
        f.write("실험이 정상적으로 완료되었습니다.\n")
        f.write("생성된 이미지: Fig_DCR_Security_Analysis.png 확인 요망.")

if __name__ == "__main__":
    main()