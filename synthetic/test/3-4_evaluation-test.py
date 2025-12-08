# =============================================================================
# [실험: 다차원 데이터 유용성 종합 평가 (All-Column Evaluation)]
# 제목: TSTR (Train Synthetic, Test Real) 확장 - 회귀 및 분류 통합 검증
# =============================================================================
#
# [예시]
# Target Column	Task Type	Metric	TRTR Score	TSTR Score	Retention(%)
# 최종낙찰금액	Regression	R2 Score	0.992	0.985	99.2%
# 최종낙찰율	Regression	R2 Score	0.850	0.820	96.4%
# 기관_상위	Classification	F1-Score	0.750	0.710	94.6%
# 업체_지역	Classification	F1-Score	0.620	0.580	93.5%
# =============================================================================
#
# [수정 이력]
# TSTR 확장 평가 및 논문용 그래프/표 자동 생성
#
# [수정 이력]
# XGBoost Invalid Classes 오류 해결 및 종합 평가
#
# [수정 사항]
# 1. Label Encoding 로직 변경: Global Encoding 대신, 학습 데이터(Synth) 기준으로 Target 재매핑
#    - XGBoost 요구사항(0~N 연속 정수) 충족
#    - Test(Real) 데이터에서 학습되지 않은 클래스(Unseen Label) 자동 필터링
# 2. 시각화 및 CSV 저장 기능 유지
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, r2_score
from sklearn.metrics import pairwise_distances
from xgboost import XGBClassifier, XGBRegressor

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

REAL_PATH = "data.csv"
SYNTH_PATH = "synthetic_data.csv"

# -----------------------------------------------------------------------------
# [1단계] 데이터 로드 및 정렬
# -----------------------------------------------------------------------------
def load_and_preprocess():
    # 1. 파일 로드
    df_synth = pd.read_csv(SYNTH_PATH, encoding='utf-8-sig')
    try:
        df_real_raw = pd.read_csv(REAL_PATH, encoding='utf-8')
    except:
        df_real_raw = pd.read_csv(REAL_PATH, encoding='cp949')

    # 2. 원본 데이터를 재현 데이터 구조로 변환
    df_real = df_real_raw.copy()
    
    # 수치형 변환
    def clean(x):
        try: return float(str(x).replace(',', ''))
        except: return np.nan
        
    for c in ['최종낙찰금액', '최종낙찰율', '기초금액']:
        if c in df_real.columns: df_real[c] = df_real[c].apply(clean)

    # 기초금액 역산
    if '기초금액' not in df_real.columns:
        if '최종낙찰금액' in df_real.columns and '최종낙찰율' in df_real.columns:
            df_real['기초금액'] = df_real['최종낙찰금액'] / (df_real['최종낙찰율'].replace(0, np.nan)/100)

    # 기관명 범주화
    if '공고기관명' in df_real.columns:
        def cat_agency(x):
            s = str(x)
            if '방위사업청' in s or '국방' in s: return '중앙행정기관'
            if any(k in s for k in ['육군', '해군', '공군', '부대']): return '군'
            return '기타'
        df_real['기관_상위'] = df_real['공고기관명'].apply(cat_agency)

    # 업체 지역 추출
    if '최종낙찰업체주소' in df_real.columns:
        df_real['업체_지역'] = df_real['최종낙찰업체주소'].astype(str).str.split(' ').str[0]

    # 3. 공통 컬럼 필터링
    target_cols = ['최종낙찰금액', '최종낙찰율', '기초금액', '기관_상위', '업체_지역']
    valid_cols = [c for c in target_cols if c in df_real.columns and c in df_synth.columns]
    
    df_real = df_real[valid_cols].dropna()
    df_synth = df_synth[valid_cols].dropna()
    
    print(f"[System] 검증 대상 컬럼: {valid_cols}")
    return df_real, df_synth

# -----------------------------------------------------------------------------
# [2단계] 종합 유용성 평가 (All-Columns Evaluation) - ★ 오류 수정됨 ★
# -----------------------------------------------------------------------------
def evaluate_all_targets(real, synth):
    print("\n" + "="*70)
    print(" [실험 1] 종합 머신러닝 유용성 평가 (All-Column TSTR)")
    print("="*70)
    
    results = []
    columns = real.columns.tolist()
    
    for target_col in columns:
        print(f"\n>> 분석 대상 타겟: [{target_col}]")
        
        # 데이터 복사
        temp_real = real.copy()
        temp_synth = synth.copy()
        
        # 1. Feature(X) 인코딩 (타겟 제외한 나머지 컬럼)
        # X는 원본/재현 합쳐서 통일된 인코딩 적용
        feature_cols = [c for c in columns if c != target_col]
        combined = pd.concat([temp_real[feature_cols], temp_synth[feature_cols]], axis=0)
        
        for col in feature_cols:
            if temp_real[col].dtype == 'object':
                le = LabelEncoder()
                le.fit(combined[col].astype(str))
                temp_real[col] = le.transform(temp_real[col].astype(str))
                temp_synth[col] = le.transform(temp_synth[col].astype(str))
        
        # X, y 분리
        X_real = temp_real.drop(columns=[target_col])
        y_real = temp_real[target_col]
        X_synth = temp_synth.drop(columns=[target_col])
        y_synth = temp_synth[target_col]
        
        # Test Set 분리 (Real Data에서만)
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42
        )
        
        # 유형 판단
        is_regression = False
        if real[target_col].dtype in ['float64', 'int64'] and real[target_col].nunique() > 20:
            is_regression = True
            model_cls = XGBRegressor
            metric_name = "R2 Score"
        else:
            is_regression = False
            model_cls = XGBClassifier
            metric_name = "F1-Score"

        # -------------------------------------------------------
        # ★ [핵심 수정] 타겟 변수(y) 인코딩 및 필터링
        # -------------------------------------------------------
        if not is_regression: # 분류 문제일 때만 적용
            # 학습 데이터(Synth)에 존재하는 클래스만으로 Encoder 학습
            le_target = LabelEncoder()
            y_synth_encoded = le_target.fit_transform(y_synth.astype(str))
            
            # Real 데이터(Train/Test)도 동일한 Encoder로 변환해야 함
            # 단, Synth에 없는 클래스가 Real에 있다면 에러 발생 -> 필터링 필요
            known_classes = set(le_target.classes_)
            
            # TRTR용 (Real Train)
            mask_train = y_train_real.astype(str).isin(known_classes)
            X_train_real = X_train_real[mask_train]
            y_train_real = le_target.transform(y_train_real[mask_train].astype(str))
            
            # Test용 (Real Test)
            mask_test = y_test_real.astype(str).isin(known_classes)
            X_test_real = X_test_real[mask_test]
            y_test_real = le_target.transform(y_test_real[mask_test].astype(str))
            
            # Synth용 (이미 인코딩됨)
            y_synth = y_synth_encoded
            
            if len(y_test_real) == 0:
                print("   [Skip] 테스트 데이터에 학습된 클래스가 없어 평가 불가.")
                continue
        # -------------------------------------------------------

        # 모델 학습
        # (A) TRTR
        model_trtr = model_cls(n_jobs=-1, random_state=42)
        model_trtr.fit(X_train_real, y_train_real)
        pred_trtr = model_trtr.predict(X_test_real)
        
        # (B) TSTR
        model_tstr = model_cls(n_jobs=-1, random_state=42)
        model_tstr.fit(X_synth, y_synth)
        pred_tstr = model_tstr.predict(X_test_real)
        
        # 점수 계산
        if is_regression:
            score_trtr = r2_score(y_test_real, pred_trtr)
            score_tstr = r2_score(y_test_real, pred_tstr)
        else:
            score_trtr = f1_score(y_test_real, pred_trtr, average='macro')
            score_tstr = f1_score(y_test_real, pred_tstr, average='macro')
            
        retention = (score_tstr / score_trtr * 100) if score_trtr > 0 else 0.0
        
        print(f"   - TRTR: {score_trtr:.4f}, TSTR: {score_tstr:.4f}, Retention: {retention:.2f}%")
        
        results.append({
            'Target Column': target_col,
            'Task Type': 'Regression' if is_regression else 'Classification',
            'Metric': metric_name,
            'TRTR Score': score_trtr,
            'TSTR Score': score_tstr,
            'Retention(%)': retention
        })

    # 결과 저장 및 시각화
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv("evaluation_all_columns.csv", index=False, encoding='utf-8-sig')
        visualize_tstr_results(res_df)
    else:
        print("[Warning] 유효한 평가 결과가 없습니다.")

def visualize_tstr_results(res_df):
    print("\n[Visualization] 실험 결과 그래프 생성 중...")
    
    # 1. 성능 비교 (Bar)
    df_melt = res_df.melt(id_vars=['Target Column'], value_vars=['TRTR Score', 'TSTR Score'], var_name='Type', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x='Target Column', y='Score', hue='Type', palette=['#1f77b4', '#ff7f0e'])
    plt.title("원본(TRTR) vs 재현(TSTR) 데이터 AI 학습 성능 비교")
    plt.ylim(0, 1.1)
    plt.savefig("Fig_TSTR_Comparison.png", dpi=300, bbox_inches='tight')
    
    # 2. 보존율 (Bar)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x='Target Column', y='Retention(%)', palette='viridis')
    plt.axhline(100, color='red', linestyle='--')
    plt.title("재현 데이터의 AI 성능 보존율 (Retention Rate)")
    plt.savefig("Fig_TSTR_Retention.png", dpi=300, bbox_inches='tight')
    
    # 3. 테이블
    plt.figure(figsize=(12, 4))
    plt.axis('off')
    table_data = res_df.copy()
    table_data['TRTR Score'] = table_data['TRTR Score'].round(3)
    table_data['TSTR Score'] = table_data['TSTR Score'].round(3)
    table_data['Retention(%)'] = table_data['Retention(%)'].round(1).astype(str) + '%'
    tbl = plt.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.2)
    plt.title("종합 유용성 평가 결과 요약")
    plt.savefig("Fig_TSTR_Table.png", dpi=300, bbox_inches='tight')
    
    print(" -> [Saved] 그래프 및 표 저장 완료")

# -----------------------------------------------------------------------------
# [3단계] 프라이버시 평가 (기존 유지)
# -----------------------------------------------------------------------------
def evaluate_privacy(real, synth):
    print("\n" + "="*70)
    print(" [실험 2] 프라이버시 안전성 평가 (DCR)")
    print("="*70)
    
    combined = pd.concat([real, synth], axis=0)
    for col in real.columns:
        if real[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(combined[col].astype(str))
            real[col] = le.transform(real[col].astype(str))
            synth[col] = le.transform(synth[col].astype(str))
            
    n = min(2000, len(real))
    real_samp = real.sample(n, random_state=42)
    synth_samp = synth.sample(n, random_state=42)
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    real_norm = scaler.fit_transform(real_samp)
    synth_norm = scaler.transform(synth_samp)
    
    dists = pairwise_distances(synth_norm, real_norm, metric='euclidean')
    dcr = dists.min(axis=1)
    
    print(f" >> 평균 거리 (Mean DCR): {np.mean(dcr):.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(dcr, kde=True, color='green', bins=50)
    plt.axvline(x=np.percentile(dcr, 5), color='red', linestyle='--', label='5th Percentile')
    plt.title("Distance to Closest Record (DCR)")
    plt.savefig("Eval_Result_DCR.png", dpi=150)
    print(" -> [Saved] Eval_Result_DCR.png")

# -----------------------------------------------------------------------------
# [Main]
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    set_korean_font()
    try:
        df_real, df_synth = load_and_preprocess()
        evaluate_all_targets(df_real.copy(), df_synth.copy())
        evaluate_privacy(df_real.copy(), df_synth.copy())
    except Exception as e:
        print(f"[Error] {e}")