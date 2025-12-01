import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from scipy import stats

from data_loader import EEGDataLoader, get_default_data_path
from analysis_utils import normalize_metadata_columns, collect_age_rt_data
from visualization_config import setup_plot_style

warnings.filterwarnings('ignore')
setup_plot_style()


def plot_age_rt_correlation(age_df, output_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(age_df['age'], age_df['response_time'], alpha=0.6, s=50, 
               color='steelblue', edgecolors='black')
    
    z = np.polyfit(age_df['age'], age_df['response_time'], 1)
    p = np.poly1d(z)
    ax.plot(age_df['age'], p(age_df['age']), "r-", linewidth=2, alpha=0.8, label='Linear fit')
    
    corr_coef, p_value = stats.pearsonr(age_df['age'], age_df['response_time'])
    
    ax.fill_between(age_df['age'].sort_values(), 
                    p(age_df['age'].sort_values()) - 100, 
                    p(age_df['age'].sort_values()) + 100, 
                    alpha=0.2, color='lightblue')
    
    ax.set_xlabel('Age (years)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Response Time (ms)', fontsize=13, fontweight='bold')
    ax.set_title(f'Correlation between Age and Response Time\nr = {corr_coef:.3f}, p = {p_value:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_coef, p_value


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
    
    if 'Age' not in metadata.columns:
        print("Error: Age information not available in metadata")
        return
    
    age_df = collect_age_rt_data(loader, metadata, real_features)
    
    if len(age_df) == 0:
        print("Warning: No data available for analysis")
        return
    
    corr_coef, p_value = plot_age_rt_correlation(age_df, 'charts/age_rt_correlation.png')
    
    print("\nAge-based correlation analysis complete")
    print(f"Correlation: r = {corr_coef:.3f}, p = {p_value:.4f}")
    print(f"Sample size: {len(age_df)} observations")
    print(f"Age range: {age_df['age'].min():.0f} - {age_df['age'].max():.0f} years")
    print(f"Mean RT: {age_df['response_time'].mean():.2f} ms")


if __name__ == "__main__":
    main()
