import numpy as np
import mne
import matplotlib.pyplot as plt
from data_loader import EEGDataLoader, get_default_data_path
import preprocessing

def compare_filters(raw, filter_configs):
    fig, axes = plt.subplots(len(filter_configs) + 1, 1, figsize=(15, 3 * (len(filter_configs) + 1)))
    
    original_data = raw.get_data(picks='eeg')
    times = raw.times
    axes[0].plot(times[:5000], original_data[0, :5000] * 1e6
    axes[0].set_title('Original (no filter)')
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].grid(True)
    
    for idx, config in enumerate(filter_configs, 1):
        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=config['l_freq'], h_freq=config['h_freq'], 
                           picks='eeg', verbose=False)
        if config.get('notch'):
            raw_filtered.notch_filter(freqs=config['notch'], picks='eeg', verbose=False)
        
        filtered_data = raw_filtered.get_data(picks='eeg')
        axes[idx].plot(times[:5000], filtered_data[0, :5000] * 1e6)
        axes[idx].set_title(f"{config['name']}: {config['l_freq']}-{config['h_freq']} Hz" + 
                          (f" + notch {config.get('notch')} Hz" if config.get('notch') else ""))
        axes[idx].set_ylabel('Amplitude (µV)')
        axes[idx].grid(True)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    return fig


def analyze_frequency_content(raw):
    psd = raw.compute_psd(fmax=100, verbose=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    psd.plot(axes=axes[0], show=False, average=True)
    axes[0].set_title('Power Spectral Density (Average across channels)')
    axes[0].set_xlim(0, 60)
    axes[0].axvline(50, color='r', linestyle='--', label='50 Hz (power line)')
    axes[0].legend()

    psds, freqs = psd.get_data(return_freqs=True)
    avg_psd = psds.mean(axis=0)
    
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-45 Hz)': (30, 45)
    }
    
    band_powers = []
    band_names = []
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        power = np.mean(avg_psd[mask])
        band_powers.append(power)
        band_names.append(name.split(' ')[0])
    
    axes[1].bar(band_names, band_powers, color='steelblue')
    axes[1].set_title('Average Power by Frequency Band')
    axes[1].set_ylabel('Power (V²/Hz)')
    axes[1].set_xlabel('Frequency Band')
    axes[1].grid(axis='y')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Loading sample data...")
    loader = EEGDataLoader(str(get_default_data_path()))
    participant_data = loader.load_participant_data(loader.participants[0])
    raw = list(participant_data.values())[0]
    
    print("\n1. Analyzing frequency content...")
    fig1 = analyze_frequency_content(raw)
    plt.savefig('results/frequency_analysis.png', dpi=150, bbox_inches='tight')
    print("   Saved: results/frequency_analysis.png")
    
    print("\n2. Comparing filter configurations...")
    filter_configs = [
        {'name': 'Current (MVP)', 'l_freq': 0.5, 'h_freq': 40, 'notch': 50},
        {'name': 'Recommended', 'l_freq': 0.1, 'h_freq': 30, 'notch': 50},
        {'name': 'Conservative', 'l_freq': 1.0, 'h_freq': 30, 'notch': 50},
        {'name': 'ERP-focused', 'l_freq': 0.1, 'h_freq': 20, 'notch': 50},
    ]
    
    fig2 = compare_filters(raw, filter_configs)
    plt.savefig('results/filter_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: results/filter_comparison.png")
    
    plt.show()
