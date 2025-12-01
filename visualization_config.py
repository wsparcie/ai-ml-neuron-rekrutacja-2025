import matplotlib.pyplot as plt
import seaborn as sns

COLOR_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F77F00',
    'danger': '#D62828',
    'honest': '#06A77D',
    'deceitful': '#D62828',
    'neutral': '#5A5A5A'
}

def setup_plot_style():
    sns.set_style('whitegrid')
    sns.set_palette('deep')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linewidth'] = 0.8
