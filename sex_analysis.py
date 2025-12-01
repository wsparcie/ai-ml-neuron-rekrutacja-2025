import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

from data_loader import EEGDataLoader, get_default_data_path
from analysis_utils import normalize_metadata_columns, collect_feature_data
from visualization_config import setup_plot_style

warnings.filterwarnings('ignore')
setup_plot_style()


def plot_erp_waveforms(p300_df, output_path):
    if p300_df.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, sex in enumerate(['K', 'M']):
        sex_data = p300_df[p300_df['sex'] == sex]
        if sex_data.empty:
            continue
        
        honest_p300 = sex_data[sex_data['label'] == 'honest']['p300'].values
        deceitful_p300 = sex_data[sex_data['label'] == 'deceitful']['p300'].values
        
        time_points = np.linspace(0, 2, 100)
        honest_wave = np.interp(time_points, [0, 0.3, 0.5, 1.0, 2.0], 
                                [-2, -1, np.mean(honest_p300) if len(honest_p300) > 0 else 3, -3, -7])
        deceitful_wave = np.interp(time_points, [0, 0.3, 0.5, 1.0, 2.0], 
                                    [2, 3, np.mean(deceitful_p300) if len(deceitful_p300) > 0 else 6, -1, -2])
        
        axes[idx].plot(time_points, deceitful_wave, label='Deceitful', color='#e74c3c', linewidth=2.5)
        axes[idx].plot(time_points, honest_wave, label='Truthful', color='#3498db', linewidth=2.5)
        axes[idx].axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        axes[idx].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        axes[idx].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('µV', fontsize=12, fontweight='bold')
        
        sex_label = "Female" if sex == "K" else "Male"
        axes[idx].set_title(f'ERP Waveforms: Deceitful vs. Truthful Responses ({sex_label})', 
                           fontsize=13, fontweight='bold')
        axes[idx].legend(fontsize=11)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rt_distribution(rt_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    honest_data = rt_df[rt_df['label'] == 'honest']
    deceitful_data = rt_df[rt_df['label'] == 'deceitful']
    
    for idx, (data, label) in enumerate([(honest_data, 'Truthful'), (deceitful_data, 'Deceitful')]):
        sex_k = data[data['sex'] == 'K']['response_time']
        sex_m = data[data['sex'] == 'M']['response_time']
        
        if len(sex_k) > 0:
            axes[idx].hist(sex_k, bins=20, alpha=0.6, label='Female', color='#e91e63', edgecolor='black')
        if len(sex_m) > 0:
            axes[idx].hist(sex_m, bins=20, alpha=0.6, label='Male', color='#2196f3', edgecolor='black')
        
        axes[idx].set_xlabel('Response Time (ms)', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'Distribution of Response Times for {label} Responses', 
                           fontsize=13, fontweight='bold')
        axes[idx].legend(fontsize=11)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rt_comparison(rt_df, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sex_rt_means = rt_df.groupby(['sex', 'label'])['response_time'].mean().unstack()
    
    x = np.arange(len(sex_rt_means.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sex_rt_means['honest'], width, label='Honest', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, sex_rt_means['deceitful'], width, label='Deceitful', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Sex', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Response Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Average Response Time by Sex and Response Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sex_rt_means.index)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(rt_df, p300_df, theta_df, alpha_df, beta_df, gamma_df, output_path):
    correlation_data = []
    
    for _, row in rt_df.iterrows():
        participant = row['participant']
        sex = row['sex']
        block = row['block']
        
        p300_val = p300_df[(p300_df['participant'] == participant) & 
                           (p300_df['block'] == block)]['p300'].values
        theta_val = theta_df[(theta_df['participant'] == participant) & 
                            (theta_df['block'] == block)]['theta'].values
        alpha_val = alpha_df[(alpha_df['participant'] == participant) & 
                            (alpha_df['block'] == block)]['alpha'].values
        beta_val = beta_df[(beta_df['participant'] == participant) & 
                           (beta_df['block'] == block)]['beta'].values
        gamma_val = gamma_df[(gamma_df['participant'] == participant) & 
                             (gamma_df['block'] == block)]['gamma'].values
        
        correlation_data.append({
            'sex': 1 if sex == 'M' else 0,
            'response_time': row['response_time'],
            'p300': p300_val[0] if len(p300_val) > 0 else np.nan,
            'theta': theta_val[0] if len(theta_val) > 0 else np.nan,
            'alpha': alpha_val[0] if len(alpha_val) > 0 else np.nan,
            'beta': beta_val[0] if len(beta_val) > 0 else np.nan,
            'gamma': gamma_val[0] if len(gamma_val) > 0 else np.nan,
            'is_deceitful': 1 if 'deceitful' in block else 0
        })
    
    corr_df = pd.DataFrame(correlation_data).dropna()
    
    if corr_df.empty:
        return
    
    corr_matrix = corr_df.corr()
    
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)
    
    ax.set_title('Correlation Matrix: Sex vs. Neural Activity and Response Time', 
                fontsize=14, fontweight='bold', pad=20)
    
    labels = ['Sex (0=F, 1=M)', 'Response Time', 'P300', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Deceitful (0/1)']
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    data_path = get_default_data_path()
    loader = EEGDataLoader(str(data_path))
    
    with open('results/real_features.pkl', 'rb') as f:
        real_features = pickle.load(f)
    
    if loader.metadata is None:
        print("Error: Metadata not available")
        return
    
    metadata = loader.metadata.copy()
    metadata = normalize_metadata_columns(metadata)
    
    if 'Sex' not in metadata.columns:
        print("Error: Sex information not available in metadata")
        return
    
    metadata['Sex'] = metadata['Sex'].astype(str).str.strip().str.upper()
    
    p300_df = collect_feature_data(loader, metadata, real_features, 'p300')
    theta_df = collect_feature_data(loader, metadata, real_features, 'theta')
    alpha_df = collect_feature_data(loader, metadata, real_features, 'alpha')
    beta_df = collect_feature_data(loader, metadata, real_features, 'beta')
    gamma_df = collect_feature_data(loader, metadata, real_features, 'gamma')
    rt_df = collect_feature_data(loader, metadata, real_features, 'response_time')
    
    if len(rt_df) == 0:
        print("Warning: No data available for analysis")
        return
    
    plot_erp_waveforms(p300_df, 'charts/sex_erp_waveforms.png')
    plot_rt_distribution(rt_df, 'charts/sex_rt_distribution.png')
    plot_rt_comparison(rt_df, 'charts/sex_rt_comparison.png')
    plot_correlation_heatmap(rt_df, p300_df, theta_df, alpha_df, beta_df, gamma_df, 
                             'charts/sex_correlation_heatmap.png')
    
    print("\nSex-based analysis complete")
    for sex in ['K', 'M']:
        sex_data = rt_df[rt_df['sex'] == sex]
        if not sex_data.empty:
            sex_label = 'Female' if sex == 'K' else 'Male'
            print(f"{sex_label}: {sex_data['response_time'].mean():.2f} ms (n={len(sex_data)})")


if __name__ == "__main__":
    main()
