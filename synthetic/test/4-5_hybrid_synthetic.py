# =============================================================================
# [최종 실험: 하이브리드 재현 및 성능 검증 (All-in-One)]
# 1. 생성: 핵심 변수(AI) + 기초금액(수식) -> 논리적 완결성 확보
# 2. 검증: 원본 vs 재현 데이터 TSTR 평가 및 시각화 즉시 출력
#
# 수정사항: 
# 1. 컬럼명 공백 제거 (strip) -> KeyError 방지
# 2. 날짜 컬럼('최종낙찰일자') 존재 여부 확인 후 처리
#
# 수정사항: 
# 1. [Critical Fix] 학습 데이터(Label) 내 NaN/Infinity 전처리 로직 강화
# 2. XGBoost 'Label contains NaN' 오류 원천 차단
#
# 수정사항: 
# 1. AI 성능 보존율(Retention Rate) 시각화 함수 추가
# 2. 결과 데이터프레임에 보존율(%) 컬럼 자동 계산 포함
#
# 수정사항: 
# 1. 최종 평가 결과(Summary)를 CSV 파일로 저장하는 기능 추가
# 2. 모든 에러 방지 로직(NaN 처리, 컬럼명 통일) 포함
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings
import os

# 머신러닝 및 평가 라이브러리
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
except ImportError:
    print("[Error] SDV 라이브러리 미설치. (!pip install sdv)")

try:
    from diffprivlib.mechanisms import Laplace
    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, f1_score
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

# [설정]
INPUT_CSV = "data.csv"
OUTPUT_SYNTHETIC = "synthetic_data_hybrid.csv"
OUTPUT_SUMMARY = "final_evaluation_summary.csv" # 결과 저장 파일명
EPSILON_DP = 0.5

# 한글 폰트 설정
def set_korean_font():
    try:
        font_paths = ['C:/Windows/Fonts/malgun.ttf', '/System/Library/Fonts/Supplemental/AppleGothic.ttf', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
        for path in font_paths:
            if os.path.exists(path):
                font_name = fm.FontProperties(fname=path).get_name()
                plt.rc('font', family=font_name)
                plt.rc('axes', unicode_minus=False)
                return
    except:
        print("[Warning] 폰트 설정 실패")

# -----------------------------------------------------------------------------
# [1단계] 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
def load_and_preprocess():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"파일이 없습니다: {INPUT_CSV}")
        
    try:
        df = pd.read_csv(INPUT_CSV, encoding='utf-8')
    except:
        df = pd.read_csv(INPUT_CSV, encoding='cp949')
    
    df.columns = df.columns.str.strip()
    print(f"[System] 데이터 로드: {len(df)}건")

    if '최종낙찰일자' in df.columns:
        df['최종낙찰일자'] = pd.to_datetime(df['최종낙찰일자'], errors='coerce')
        df['최종낙찰일자_year'] = df['최종낙찰일자'].dt.year
        df['최종낙찰일자_month'] = df['최종낙찰일자'].dt.month

    def clean(x):
        try: return float(str(x).replace(',', ''))
        except: return np.nan
    
    for c in ['최종낙찰금액', '최종낙찰율', '기초금액']:
        if c in df.columns: df[c] = df[c].apply(clean)

    if '기초금액' not in df.columns or df['기초금액'].isnull().mean() > 0.5:
        if '최종낙찰금액' in df.columns and '최종낙찰율' in df.columns:
            rate = df['최종낙찰율'].replace(0, np.nan)
            df['기초금액'] = df['최종낙찰금액'] / (rate / 100)
    
    if '공고기관명' in df.columns:
        def cat_agency(x):
            s = str(x)
            if '방위사업청' in s or '국방' in s: return '중앙행정기관'
            if any(k in s for k in ['육군', '해군', '공군']): return '군'
            return '기타'
        df['기관_상위'] = df['공고기관명'].apply(cat_agency)
    else:
        df['기관_상위'] = '기타'

    if '최종낙찰금액' in df.columns:
        df['최종낙찰금액_log'] = np.log1p(df['최종낙찰금액'].clip(lower=0))
    
    important_cols = ['최종낙찰금액', '최종낙찰율', '기초금액', '기관_상위']
    important_cols = [c for c in important_cols if c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=important_cols)
    
    return df

# -----------------------------------------------------------------------------
# [2단계] 하이브리드 생성 (AI + Rule)
# -----------------------------------------------------------------------------
def generate_hybrid(df):
    print("\n[Generation] 하이브리드 재현 데이터 생성 시작...")
    
    candidates = ['최종낙찰금액_log', '최종낙찰율', '기관_상위', '최종낙찰일자_year', '최종낙찰일자_month']
    train_cols = [c for c in candidates if c in df.columns]
    df_train = df[train_cols].dropna()
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    
    # 모델 학습 (속도를 위해 Epochs=300 설정, 실제 실험용은 500 권장)
    model = CTGANSynthesizer(metadata, epochs=500, verbose=True)
    model.fit(df_train)
    
    synth = model.sample(len(df))
    
    if '최종낙찰금액_log' in synth.columns:
        synth['최종낙찰금액'] = np.expm1(synth['최종낙찰금액_log']).clip(lower=0)
    
    if '최종낙찰금액' in synth.columns and '최종낙찰율' in synth.columns:
        print("[Post-processing] 수식 기반 '기초금액' 복원 중...")
        rate = synth['최종낙찰율'].replace(0, 0.1) 
        synth['기초금액'] = synth['최종낙찰금액'] / (rate / 100)
        synth['기초금액'] = synth['기초금액'].replace([np.inf, -np.inf], 0).fillna(0).clip(lower=0)
    
    return synth

# -----------------------------------------------------------------------------
# [3단계] 성능 평가 (TSTR)
# -----------------------------------------------------------------------------
def evaluate_tstr(real, synth):
    print("\n" + "="*60)
    print(" [Evaluation] TSTR 성능 평가")
    print("="*60)
    
    targets = ['기초금액', '최종낙찰금액', '최종낙찰율', '기관_상위']
    results = []
    
    for target in targets:
        if target not in real.columns or target not in synth.columns: continue
        
        real_clean = real.dropna(subset=[target])
        synth_clean = synth.dropna(subset=[target])
        
        if len(real_clean) == 0 or len(synth_clean) == 0: continue

        drop_cols = [target] + [c for c in real.columns if 'log' in c or '공고' in c or '일자' in c]
        X_real = real_clean.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
        y_real = real_clean[target]
        X_synth = synth_clean.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
        y_synth = synth_clean[target]
        
        if X_real.shape[1] == 0: continue

        is_classification = False
        if y_real.dtype == 'object' or y_real.nunique() < 20:
            is_classification = True
            le = LabelEncoder()
            y_synth = le.fit_transform(y_synth.astype(str))
            mask = y_real.astype(str).isin(le.classes_)
            if mask.sum() == 0: continue 
            X_real = X_real[mask]
            y_real = le.transform(y_real[mask].astype(str))
            model = XGBClassifier(n_jobs=-1, random_state=42)
            metric = "F1-Score"
        else:
            if not np.isfinite(y_real).all() or not np.isfinite(y_synth).all(): continue
            model = XGBRegressor(n_jobs=-1, random_state=42)
            metric = "R2 Score"
            
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
            
            model.fit(X_train, y_train)
            pred_trtr = model.predict(X_test)
            
            model.fit(X_synth, y_synth)
            pred_tstr = model.predict(X_test)
            
            if is_classification:
                s_trtr = f1_score(y_test, pred_trtr, average='macro')
                s_tstr = f1_score(y_test, pred_tstr, average='macro')
            else:
                s_trtr = r2_score(y_test, pred_trtr)
                s_tstr = r2_score(y_test, pred_tstr)
                
            retention = (s_tstr / s_trtr) * 100 if s_trtr > 0 else 0
            
            print(f" >> 타겟: {target} | TRTR: {s_trtr:.4f} | TSTR: {s_tstr:.4f} | 보존율: {retention:.1f}%")
            results.append({'Target': target, 'Metric': metric, 'TRTR': s_trtr, 'TSTR': s_tstr, 'Retention(%)': retention})
            
        except Exception as e:
            print(f"   [Error] {target} 학습 오류: {e}")

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# [4단계] 시각화
# -----------------------------------------------------------------------------
def plot_results(real, synth, res_df):
    if '기초금액' in real.columns and '최종낙찰금액' in real.columns:
        plt.figure(figsize=(12, 5))
        n = min(1000, len(real))
        
        plt.subplot(1, 2, 1)
        samp = real.sample(n)
        sns.scatterplot(x=samp['기초금액'], y=samp['최종낙찰금액'], alpha=0.5)
        plt.title("원본 데이터 (Ground Truth)")
        
        plt.subplot(1, 2, 2)
        samp = synth.sample(n)
        sns.scatterplot(x=samp['기초금액'], y=samp['최종낙찰금액'], color='orange', alpha=0.5)
        plt.title("하이브리드 재현 데이터 (Hybrid Synthetic)")
        
        plt.tight_layout()
        plt.savefig("Hybrid_Scatter_Final.png")
        print("\n[Visualization] 산점도 저장 완료: Hybrid_Scatter_Final.png")

    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        colors = ['green' if x >= 90 else '#1f77b4' for x in res_df['Retention(%)']]
        ax = sns.barplot(data=res_df, x='Target', y='Retention(%)', palette=colors)
        plt.axhline(100, color='red', linestyle='--', label='Ideal (100%)')
        
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        plt.title("Hybrid 재현 데이터 AI 성능 보존율 (Retention Rate)", fontsize=15)
        plt.ylabel("보존율 (%)", fontsize=12)
        plt.ylim(0, max(120, res_df['Retention(%)'].max() + 10))
        plt.legend()
        plt.savefig("Hybrid_Retention_Rate.png")
        print("[Visualization] 보존율 그래프 저장 완료: Hybrid_Retention_Rate.png")

# -----------------------------------------------------------------------------
# [Main]
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    set_korean_font()
    
    try:
        # 1. 전처리
        df_real = load_and_preprocess()
        
        # 2. 생성
        df_synth = generate_hybrid(df_real)
        df_synth.to_csv(OUTPUT_SYNTHETIC, index=False, encoding='utf-8-sig')
        print(f"[System] 파일 저장 완료: {OUTPUT_SYNTHETIC}")
        
        # 3. 평가
        res_df = evaluate_tstr(df_real, df_synth)
        
        # 4. 시각화
        plot_results(df_real, df_synth, res_df)
        
        # 5. [추가] 결과 요약 CSV 저장
        print("\n[Final Summary]")
        print(res_df)
        res_df.to_csv(OUTPUT_SUMMARY, index=False, encoding='utf-8-sig')
        print(f"[System] 평가 결과 저장 완료: {OUTPUT_SUMMARY}")
        
    except Exception as e:
        print(f"[Critical Error] {e}")