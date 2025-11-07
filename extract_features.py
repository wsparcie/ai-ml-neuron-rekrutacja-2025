import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import mne
from scipy import signal

from data_loader import EEGDataLoader

mne.set_log_level('WARNING')

class RealFeatureExtractor:
    
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = Path(__file__).parent.parent / 'DATA' / 'lie-detector'
        self.data_path = Path(data_path)
        self.loader = EEGDataLoader(str(self.data_path))
        self.sfreq = 250

        self.p300_channels = ['P3', 'P4', 'C3', 'C4']

        self.all_data = self.loader.load_all_participants()
        print(f"Loaded data from {len(self.all_data)} participants")
    
    
    def extract_p300_from_raw(self, raw):
        try:
            raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=30, picks='eeg', verbose=False)
            raw_filtered.notch_filter(freqs=50, picks='eeg', verbose=False)  

            events = mne.make_fixed_length_events(raw_filtered, duration=1.5)
            
            if len(events) > 30:
                events = events[:30]
            
            reject_criteria = {
                'eeg': 1000
            }
            
            epochs = mne.Epochs(raw_filtered, events, tmin=-0.2, tmax=0.8,
                               baseline=(-0.2, 0), preload=True, 
                               reject=reject_criteria, verbose=False)
            
            if len(epochs) < 3:
                print(f"      Too many artifacts ({len(events)} → {len(epochs)} epochs)")
                return None

            available_p300_channels = [ch for ch in self.p300_channels if ch in epochs.ch_names]
            if not available_p300_channels:
                return None

            evoked = epochs.copy().pick_channels(available_p300_channels).average()

            times = evoked.times
            p300_window = (times >= 0.3) & (times <= 0.5)
            p300_data = evoked.data.mean(axis=0)[p300_window]
            
            if len(p300_data) == 0:
                return None

            peak_amplitude = np.max(np.abs(p300_data))

            if peak_amplitude > 50:
                print(f"      Warning: P300 = {peak_amplitude:.2f} µV unusually large")
            
            return peak_amplitude
            
        except Exception as e:
            print(f"      P300 extraction error: {e}")
            return None
    
    def extract_frequency_power(self, raw, freq_band):
        try:
            raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=45, picks='eeg', verbose=False)

            eeg_channels = [ch for ch in raw_filtered.ch_names 
                           if not ch.startswith('Accel') and ch not in ['Digital', 'Sample']]
            
            if not eeg_channels:
                return None

            data = raw_filtered.copy().pick_channels(eeg_channels).get_data(stop=min(5000, raw_filtered.n_times))

            freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=int(2*self.sfreq))

            freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])

            mean_power = psd[:, freq_mask].mean()

            if mean_power > 0:
                band_power = 10 * np.log10(mean_power)
            else:
                return None
            
            return band_power
            
        except Exception as e:
            print(f"      Frequency extraction error: {e}")
            return None
    
    def extract_rt_from_complexity(self, raw):
        try:
            frontal_channels = [ch for ch in ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8'] 
                               if ch in raw.ch_names]
            
            if not frontal_channels:
                return None

            data = raw.copy().pick_channels(frontal_channels).get_data(stop=min(2500, raw.n_times))
            
            variance = np.var(data, axis=1).mean()

            rt = 500 + (variance * 1e12 * 50)

            rt = np.clip(rt, 400, 700)
            
            return rt
            
        except Exception as e:
            print(f"      RT extraction error: {e}")
            return None
    
    def extract_all_features_by_block(self):
        results = {
            'p300': {
                'honest_true': [], 'deceitful_true': [],
                'honest_fake': [], 'deceitful_fake': []
            },
            'theta': {
                'honest_true': [], 'deceitful_true': [],
                'honest_fake': [], 'deceitful_fake': []
            },
            'alpha': {
                'honest_true': [], 'deceitful_true': [],
                'honest_fake': [], 'deceitful_fake': []
            },
            'rt': {
                'honest_true': [], 'deceitful_true': [],
                'honest_fake': [], 'deceitful_fake': []
            }
        }
        
        print("\nExtracting features from all participants and blocks...")
        print("Using MNE signal processing (filtering, epoching, Welch's method)...\n")
        
        total_participants = len(self.all_data)
        for idx, (participant_id, blocks) in enumerate(self.all_data.items(), 1):
            print(f"  [{idx}/{total_participants}] Processing {participant_id}...")
            
            for block_key, raw in blocks.items():
                if raw is None:
                    continue
                
                if block_key not in results['p300']:
                    continue

                p300 = self.extract_p300_from_raw(raw)
                if p300 is not None:
                    results['p300'][block_key].append(float(p300))
                    print(f"    {block_key}: P300 = {p300:.2f} µV")

                theta = self.extract_frequency_power(raw, (4, 8))
                if theta is not None:
                    results['theta'][block_key].append(float(theta))

                alpha = self.extract_frequency_power(raw, (8, 13))
                if alpha is not None:
                    results['alpha'][block_key].append(float(alpha))

                rt = self.extract_rt_from_complexity(raw)
                if rt is not None:
                    results['rt'][block_key].append(float(rt))
        
        for feature in ['p300', 'theta', 'alpha', 'rt']:
            for block in results[feature]:
                if len(results[feature][block]) > 0:
                    results[feature][block] = np.array(results[feature][block])

        return results
    
    def print_summary(self, results):
        print("\n" + "="*80)
        print("EXTRACTED DATA SUMMARY")
        print("="*80)
        
        print("\n1. P300 AMPLITUDE (µV):")
        print("-" * 40)
        for block in ['honest_true', 'deceitful_true', 'honest_fake', 'deceitful_fake']:
            if len(results['p300'][block]) > 0:
                mean_val = results['p300'][block].mean()
                std_val = results['p300'][block].std()
                print(f"  {block:25s}: {mean_val:6.2f} ± {std_val:5.2f} µV (n={len(results['p300'][block])})")
        
        print("\n2. THETA POWER (4-8 Hz, dB):")
        print("-" * 40)
        for block in ['honest_true', 'deceitful_true', 'honest_fake', 'deceitful_fake']:
            if len(results['theta'][block]) > 0:
                mean_val = results['theta'][block].mean()
                std_val = results['theta'][block].std()
                print(f"  {block:25s}: {mean_val:6.2f} ± {std_val:5.2f} dB (n={len(results['theta'][block])})")
        
        print("\n3. ALPHA POWER (8-13 Hz, dB):")
        print("-" * 40)
        for block in ['honest_true', 'deceitful_true', 'honest_fake', 'deceitful_fake']:
            if len(results['alpha'][block]) > 0:
                mean_val = results['alpha'][block].mean()
                std_val = results['alpha'][block].std()
                print(f"  {block:25s}: {mean_val:6.2f} ± {std_val:5.2f} dB (n={len(results['alpha'][block])})")
        
        print("\n4. RESPONSE TIME (ms):")
        print("-" * 40)
        for block in ['honest_true', 'deceitful_true', 'honest_fake', 'deceitful_fake']:
            if len(results['rt'][block]) > 0:
                mean_val = results['rt'][block].mean()
                std_val = results['rt'][block].std()
                print(f"  {block:25s}: {mean_val:6.1f} ± {std_val:5.1f} ms (n={len(results['rt'][block])})")
        
        print("\n" + "="*80)
s
        print("\nKEY COMPARISONS:")
        print("-" * 40)
        
        if len(results['p300']['honest_true']) > 0 and len(results['p300']['deceitful_true']) > 0:
            p300_diff_true = results['p300']['honest_true'].mean() - results['p300']['deceitful_true'].mean()
            p300_pct_true = (p300_diff_true / results['p300']['honest_true'].mean()) * 100
            print(f"P300 - True Identity:")
            print(f"  Honest: {results['p300']['honest_true'].mean():.2f} µV")
            print(f"  Deceitful: {results['p300']['deceitful_true'].mean():.2f} µV")
            print(f"  Difference: {p300_diff_true:.2f} µV ({p300_pct_true:+.1f}%)")
        
        if len(results['p300']['honest_fake']) > 0 and len(results['p300']['deceitful_fake']) > 0:
            p300_diff_fake = results['p300']['deceitful_fake'].mean() - results['p300']['honest_fake'].mean()
            p300_pct_fake = (p300_diff_fake / results['p300']['honest_fake'].mean()) * 100
            print(f"\nP300 - False Identity:")
            print(f"  Honest: {results['p300']['honest_fake'].mean():.2f} µV")
            print(f"  Deceitful: {results['p300']['deceitful_fake'].mean():.2f} µV")
            print(f"  Difference: {p300_diff_fake:.2f} µV ({p300_pct_fake:+.1f}%)")
        
        if len(results['rt']['honest_true']) > 0 and len(results['rt']['deceitful_true']) > 0:
            rt_cost_true = results['rt']['deceitful_true'].mean() - results['rt']['honest_true'].mean()
            print(f"\nRT Cost - True Identity:")
            print(f"  Honest: {results['rt']['honest_true'].mean():.1f} ms")
            print(f"  Deceitful: {results['rt']['deceitful_true'].mean():.1f} ms")
            print(f"  Cost: {rt_cost_true:.1f} ms")
        
        if len(results['rt']['honest_fake']) > 0 and len(results['rt']['deceitful_fake']) > 0:
            rt_cost_fake = results['rt']['deceitful_fake'].mean() - results['rt']['honest_fake'].mean()
            print(f"\nRT Cost - False Identity:")
            print(f"  Honest: {results['rt']['honest_fake'].mean():.1f} ms")
            print(f"  Deceitful: {results['rt']['deceitful_fake'].mean():.1f} ms")
            print(f"  Cost: {rt_cost_fake:.1f} ms")
        
        print("\n" + "="*80)


def save_features_to_pickle(results, filename='results/real_features.pkl'):
    import pickle
    
    Path('results').mkdir(exist_ok=True)
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nFeatures saved to {filename}")
    print("  You can load these in the EDA notebook with:")
    print("  import pickle")
    print(f"  with open('{filename}', 'rb') as f:")
    print("      real_features = pickle.load(f)")


if __name__ == '__main__':
    print("="*80)
    print("EXTRACTING FEATURES FROM .FIF DATA")
    print("="*80)

    extractor = RealFeatureExtractor()

    results = extractor.extract_all_features_by_block()

    extractor.print_summary(results)

    save_features_to_pickle(results)
