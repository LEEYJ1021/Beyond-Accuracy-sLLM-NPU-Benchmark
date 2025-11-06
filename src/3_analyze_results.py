# src/3_analyze_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import patsy
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_1samp
import warnings
import platform
import os

# ======================================================================
# == CONFIGURATION                                                    ==
# ======================================================================
warnings.filterwarnings('ignore')

# --- File Paths ---
# Assumes the script is run from the root directory of the project
DATA_DIR = "data"
RESULTS_DIR = "results"
PATH_MAIN = os.path.join(DATA_DIR, "Data_정제.xlsx")
PATH_HF = os.path.join(DATA_DIR, "Data_허깅페이스.xlsx")
PATH_TS = os.path.join(DATA_DIR, "Data_시계열.xlsx")

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Font Settings for Plots ---
def setup_fonts():
    """Sets up matplotlib fonts for Korean and English."""
    system = platform.system()
    font_name = None
    if system == "Windows":
        font_path = "C:/Windows/Fonts/malgun.ttf"
        if os.path.exists(font_path):
            font_name = fm.FontProperties(fname=font_path).get_name()
    elif system == "Darwin": # macOS
        font_path = "/System/Library/Fonts/AppleGothic.ttf"
        if os.path.exists(font_path):
            font_name = fm.FontProperties(fname=font_path).get_name()
    else: # Linux
        # Try common Korean fonts
        for path in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
            if os.path.exists(path):
                font_name = fm.FontProperties(fname=path).get_name()
                break
    
    if font_name:
        rc('font', family=font_name)
        sns.set(font=font_name)
        print(f"✅ Font '{font_name}' applied for plots.")
    else:
        print("⚠️ Korean font not found. Plots may not display Korean characters correctly.")
        
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", context="talk", palette="viridis")

# ======================================================================
# == DATA LOADING AND PREPROCESSING (STEP 0 & 1)                      ==
# ======================================================================
def load_and_preprocess_data():
    """Loads data from Excel files and performs extensive preprocessing and merging."""
    print("\n" + "="*80)
    print("🔧 STEP 0 & 1: Loading and Preprocessing Data")
    print("="*80)
    
    try:
        df_main_raw = pd.read_excel(PATH_MAIN)
        df_hf_raw = pd.read_excel(PATH_HF)
        df_ts_raw = pd.read_excel(PATH_TS)
        print("✅ All data files loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Data file not found. Make sure files are in the '{DATA_DIR}' directory.")
        print(f"   - Details: {e}")
        return None, None

    # Standardize column names
    def standardize_columns(df):
        df.columns = (df.columns
                     .str.strip()
                     .str.lower()
                     .str.replace(r'[\s\(\)-]+', '_', regex=True)
                     .str.replace(r'_b$', '_b', regex=True)
                     .str.replace(r'%', '_pct')
                     .str.replace(r'/', '_per_'))
        return df

    df_main_p = standardize_columns(df_main_raw)
    df_ts_p = standardize_columns(df_ts_raw)

    # Process and merge Hugging Face metadata
    df_hf_T = df_hf_raw.set_index('항목').T.reset_index().rename(columns={'index': 'model'})
    df_hf_T = standardize_columns(df_hf_T)
    
    df_main_p['model_norm'] = df_main_p['model'].str.lower().str.strip()
    df_hf_T['model_norm'] = df_hf_T['model'].str.lower().str.strip()
    df_merged = pd.merge(df_main_p, df_hf_T, on='model_norm', how='left', suffixes=('', '_hf'))
    df_merged.drop(columns=['model_norm', 'model_hf'], inplace=True, errors='ignore')

    # Clean time-series data
    ts_numeric_cols = ['latency_ms', 'tokens_generated', 'tokens_per_sec']
    for col in ts_numeric_cols:
        df_ts_p[col] = pd.to_numeric(df_ts_p[col], errors='coerce')
    
    df_ts_p.dropna(subset=['latency_ms'], inplace=True)
    print(f"   - Merged main data shape: {df_merged.shape}")
    print(f"   - Cleaned time-series data shape: {df_ts_p.shape}")
    
    return df_merged, df_ts_p

# ======================================================================
# == HYPOTHESIS TESTING AND ANALYSIS                                  ==
# ======================================================================

def analyze_h1_interaction_effect(df_ts, df_main):
    """(H1) Test for interaction between model size and task complexity on latency."""
    print("\n" + "="*80)
    print("🔬 HYPOTHESIS 1: Interaction of Model Size and Task on Latency")
    print("="*80)
    
    model_params = df_main[['model', 'parameter_b']].drop_duplicates()
    df_hypo1 = pd.merge(df_ts, model_params, on='model', how='inner')
    df_hypo1.dropna(subset=['latency_ms', 'task', 'parameter_b'], inplace=True)

    if df_hypo1['task'].nunique() < 2 or df_hypo1['parameter_b'].nunique() < 2:
        print("   - ⚠️ Insufficient data to test for interaction.")
        return

    # Two-way ANOVA
    model_formula = 'latency_ms ~ C(task) * parameter_b'
    model = ols(model_formula, data=df_hypo1).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    p_interaction = anova_table.loc['C(task):parameter_b', 'PR(>F)']
    ss_interaction = anova_table.loc['C(task):parameter_b', 'sum_sq']
    ss_residual = anova_table.loc['Residual', 'sum_sq']
    partial_eta_sq = ss_interaction / (ss_interaction + ss_residual)

    print("📊 ANOVA Results:")
    print(anova_table)
    print("\n" + "-"*50)
    print(f"   - Interaction Effect (task * size) p-value: {p_interaction:.4f}")
    print(f"   - Partial Eta-Squared (η²p): {partial_eta_sq:.4f}")
    print(f"   - Conclusion: {'A significant interaction effect was found.' if p_interaction < 0.05 else 'No significant interaction effect.'}")
    print("-" * 50)

    # Visualization
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_hypo1, x='parameter_b', y='latency_ms', hue='task', marker='o', errorbar='ci')
    plt.title('Interaction Effect of Model Size and Task on Inference Latency', fontsize=16, fontweight='bold')
    plt.xlabel('Model Size (Billion Parameters)')
    plt.ylabel('Inference Latency (ms)')
    plt.legend(title='Task')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(RESULTS_DIR, "H1_interaction_plot.png"), bbox_inches='tight')
    plt.show()

def analyze_h2_nonlinear_context_latency(df_ts, df_main):
    """(H2) Test for a non-linear relationship between context length and latency."""
    print("\n" + "="*80)
    print("🔬 HYPOTHESIS 2: Non-linear Relationship between Context Length and Latency")
    print("="*80)
    
    lat_mean = df_ts.groupby('model')['latency_ms'].mean().reset_index()
    ctx = df_main[['model', 'context_length']].drop_duplicates()
    ctx['context_length'] = pd.to_numeric(ctx['context_length'], errors='coerce')
    df_hypo2 = pd.merge(lat_mean, ctx, on='model', how='inner').dropna()

    if len(df_hypo2) < 6:
        print("   - ⚠️ Insufficient data for spline regression.")
        return

    y, X = patsy.dmatrices('latency_ms ~ context_length', data=df_hypo2, return_type='dataframe')
    X_spline = patsy.dmatrix("bs(context_length, df=4, degree=3)", data=df_hypo2, return_type='dataframe')

    # Models
    linear_model = sm.OLS(y, X).fit()
    spline_model = sm.OLS(y, X_spline).fit()

    delta_r2 = spline_model.rsquared_adj - linear_model.rsquared_adj
    
    print("📊 Model Comparison:")
    print(f"   - Linear Model Adj. R²: {linear_model.rsquared_adj:.4f}, AIC: {linear_model.aic:.2f}")
    print(f"   - Spline Model Adj. R²: {spline_model.rsquared_adj:.4f}, AIC: {spline_model.aic:.2f}")
    print("\n" + "-"*50)
    print(f"   - Improvement in Explained Variance (ΔR²): {delta_r2:.4f}")
    print(f"   - Conclusion: {'A non-linear relationship is strongly supported.' if delta_r2 > 0.1 else 'A non-linear relationship is suggested.'}")
    print("-" * 50)

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.scatter(df_hypo2['context_length'], df_hypo2['latency_ms'], label='Actual Data', alpha=0.7, s=80)
    
    sorted_idx = df_hypo2['context_length'].argsort()
    plt.plot(df_hypo2['context_length'][sorted_idx], linear_model.predict()[sorted_idx], 'r--', label=f'Linear Fit (R²={linear_model.rsquared_adj:.2f})')
    plt.plot(df_hypo2['context_length'][sorted_idx], spline_model.predict()[sorted_idx], 'g-', label=f'Spline Fit (R²={spline_model.rsquared_adj:.2f})', linewidth=2)
    
    plt.title('Context Length vs. Mean Latency: Linear vs. Non-linear Fit', fontsize=16, fontweight='bold')
    plt.xlabel('Maximum Context Length')
    plt.ylabel('Mean Inference Latency (ms)')
    plt.legend()
    plt.xscale('log')
    plt.savefig(os.path.join(RESULTS_DIR, "H2_spline_plot.png"), bbox_inches='tight')
    plt.show()

def analyze_h3_latency_variance(df_ts):
    """(H3) Test for heterogeneity of latency variance across tasks."""
    print("\n" + "="*80)
    print("🔬 HYPOTHESIS 3: Heterogeneity of Latency Variance Across Tasks")
    print("="*80)

    if 'task' not in df_ts.columns or df_ts['task'].nunique() < 2:
        print("   - ⚠️ Insufficient task data for variance analysis.")
        return

    tasks = df_ts['task'].unique()
    groups = [df_ts['latency_ms'][df_ts['task'] == task].dropna() for task in tasks]
    
    # Levene's test for homogeneity of variances
    levene_stat, levene_p = stats.levene(*groups, center='median')
    
    print("📊 Levene's Test Results:")
    print(f"   - W-statistic: {levene_stat:.4f}")
    print(f"   - p-value: {levene_p:.4f}")
    print(f"   - Conclusion: {'Latency variance is significantly different across tasks.' if levene_p < 0.05 else 'No significant difference in latency variance across tasks.'}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_ts, x='task', y='latency_ms')
    plt.title('Latency Distribution Across Different Tasks', fontsize=16, fontweight='bold')
    plt.xlabel('Task')
    plt.ylabel('Inference Latency (ms)')
    plt.yscale('log')
    plt.savefig(os.path.join(RESULTS_DIR, "H3_variance_plot.png"), bbox_inches='tight')
    plt.show()

def analyze_h4_throughput_drivers(df_main):
    """(H4) Identify key drivers of token throughput using a Random Forest model."""
    print("\n" + "="*80)
    print("🔬 HYPOTHESIS 4: Key Drivers of Token Throughput")
    print("="*80)

    features_to_use = ['parameter_b', 'peak_npu_mem_gb', 'latency_avg_ms', 'context_length']
    target = 'throughput_tokens_per_sec'
    
    # Rename columns to be consistent
    df_hypo4 = df_main.rename(columns={'peak_npu_mem_휴': 'peak_npu_mem_gb', 'latency_avg_ms_': 'latency_avg_ms'})
    
    df_hypo4 = df_hypo4[features_to_use + [target]].dropna()
    for col in df_hypo4.columns:
        df_hypo4[col] = pd.to_numeric(df_hypo4[col], errors='coerce')
    df_hypo4.dropna(inplace=True)

    if len(df_hypo4) < 5:
        print("   - ⚠️ Insufficient data for feature importance analysis.")
        return

    X = df_hypo4[features_to_use]
    y = df_hypo4[target]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    result = permutation_importance(rf, X, y, n_repeats=30, random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()

    importance_df = pd.DataFrame({
        'feature': X.columns[perm_sorted_idx],
        'importance': result.importances_mean[perm_sorted_idx]
    }).sort_values('importance', ascending=False)
    
    print("📊 Permutation Importance Results:")
    print(importance_df)
    print("\n" + "-"*50)
    print(f"   - Most Important Feature: {importance_df.iloc[0]['feature']}")
    print("-" * 50)

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance for Predicting Token Throughput", fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(RESULTS_DIR, "H4_feature_importance.png"), bbox_inches='tight')
    plt.show()

def analyze_h5_latency_distribution(df_ts):
    """(H5) Analyze latency distributions for non-normality and stability."""
    print("\n" + "="*80)
    print("🔬 HYPOTHESIS 5: Characterizing Latency Distributions")
    print("="*80)

    stability_metrics = []
    for model_name, group_data in df_ts.groupby('model'):
        latencies = group_data['latency_ms'].dropna()
        if len(latencies) < 20:
            continue
        
        # Normality Test
        shapiro_stat, shapiro_p = stats.shapiro(latencies)
        
        # Stability Metrics
        p50 = latencies.quantile(0.50)
        p95 = latencies.quantile(0.95)
        cv = latencies.std() / latencies.mean() if latencies.mean() > 0 else 0
        
        q1 = latencies.quantile(0.25)
        q3 = latencies.quantile(0.75)
        iqr = q3 - q1
        outliers = latencies[(latencies < q1 - 1.5 * iqr) | (latencies > q3 + 1.5 * iqr)]
        outlier_ratio = len(outliers) / len(latencies)
        
        metrics = {
            'model': model_name,
            'is_normal': shapiro_p > 0.05,
            'cv': cv,
            'p95_p50_ratio': p95 / p50 if p50 > 0 else 1,
            'outlier_ratio': outlier_ratio
        }
        stability_metrics.append(metrics)

    df_stability = pd.DataFrame(stability_metrics)
    
    # Calculate Optimization Priority Score
    df_stability['opt_priority'] = (
        df_stability['outlier_ratio'] * 0.3 +
        (df_stability['p95_p50_ratio'] - 1).clip(0) * 0.4 +
        df_stability['cv'] * 0.3
    )

    print("📊 Distribution Analysis Summary:")
    print(f"   - Models rejecting normality (p<0.05): {len(df_stability[~df_stability['is_normal']])}/{len(df_stability)}")
    
    # Test if key metrics are significantly above stability thresholds
    cv_test = ttest_1samp(df_stability['cv'], 0.1, alternative='greater')
    tail_test = ttest_1samp(df_stability['p95_p50_ratio'], 1.2, alternative='greater')

    print(f"   - Mean CV ({df_stability['cv'].mean():.3f}) is significantly > 0.1: {cv_test.pvalue < 0.05} (p={cv_test.pvalue:.4f})")
    print(f"   - Mean Tail Ratio ({df_stability['p95_p50_ratio'].mean():.3f}) is significantly > 1.2: {tail_test.pvalue < 0.05} (p={tail_test.pvalue:.4f})")
    
    df_sorted = df_stability.sort_values('opt_priority', ascending=False)
    print("\n" + "-"*50)
    print("🏆 Top 5 Models Requiring Optimization:")
    print(df_sorted.head())
    print("-" * 50)

    # Visualization: Distributional Cost Map
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_stability, x='cv', y='p95_p50_ratio', size='outlier_ratio', hue='opt_priority',
                    sizes=(50, 500), palette='coolwarm_r', legend='auto')
    
    for i, row in df_stability.iterrows():
        plt.text(row['cv'] + 0.01, row['p95_p50_ratio'], row['model'], fontsize=9)

    plt.title('Distributional Cost Map: Speed vs. Stability Trade-off', fontsize=16, fontweight='bold')
    plt.xlabel('Variability (Coefficient of Variation)')
    plt.ylabel('Tail Risk (P95/P50 Ratio)')
    plt.legend(title='Opt. Priority', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(RESULTS_DIR, "H5_cost_map.png"), bbox_inches='tight')
    plt.show()

# ======================================================================
# == MAIN EXECUTION                                                   ==
# ======================================================================
if __name__ == "__main__":
    setup_fonts()
    
    # Load and preprocess data
    df_main, df_ts = load_and_preprocess_data()
    
    if df_main is not None and df_ts is not None:
        # Run all hypothesis tests
        analyze_h1_interaction_effect(df_ts, df_main)
        analyze_h2_nonlinear_context_latency(df_ts, df_main)
        analyze_h3_latency_variance(df_ts)
        analyze_h4_throughput_drivers(df_main)
        analyze_h5_latency_distribution(df_ts)
        
        print("\n\n" + "="*80)
        print("🎉 All analyses completed successfully!")
        print(f"   - Check the '{RESULTS_DIR}' directory for saved plots.")
        print("="*80)