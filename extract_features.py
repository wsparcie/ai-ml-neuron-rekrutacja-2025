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
            
            reject_criteria = {'eeg': 1000}
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
            
            data = raw_filtered.copy().pick_channels(eeg_channels).get_data()
            
            freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=int(2 * self.sfreq))
            freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
            mean_power = psd[:, freq_mask].mean()
            
            if mean_power <= 0:
                return None
            
            return 10 * np.log10(mean_power)
        
        except Exception as e:
            print(f"      Frequency extraction error: {e}")
            return None
    
    def extract_gamma_power(self, raw):
        try:
            raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=100, picks='eeg', verbose=False)
            
            raw_filtered.notch_filter(freqs=[50, 100], picks='eeg', verbose=False)
            
            eeg_channels = [ch for ch in raw_filtered.ch_names 
                           if not ch.startswith('Accel') and ch not in ['Digital', 'Sample']]
            
            if not eeg_channels:
                return None
            
            data = raw_filtered.copy().pick_channels(eeg_channels).get_data()
            
            freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=int(2 * self.sfreq))
            
            gamma_mask = (freqs >= 30) & (freqs <= 100)
            notch_mask_50 = (freqs >= 48) & (freqs <= 52)
            notch_mask_100 = (freqs >= 98) & (freqs <= 102)
            
            valid_gamma = gamma_mask & ~notch_mask_50 & ~notch_mask_100
            
            if valid_gamma.sum() == 0:
                return None
            
            mean_power = psd[:, valid_gamma].mean()
            
            if mean_power <= 0:
                return None
            
            return 10 * np.log10(mean_power)
        
        except Exception as e:
            print(f"      Gamma extraction error: {e}")
            return None
    
    def extract_rt_from_annotations(self, raw):
        try:
            annotations = raw.annotations
            
            if len(annotations) == 0:
                return None
            
            onsets = annotations.onset
            descriptions = annotations.description
            
            question_indices = []
            response_indices = []
            
            for i, desc in enumerate(descriptions):
                if 'PersonalDataField' in desc:
                    question_indices.append(i)
                elif 'ParticipantResponse' in desc:
                    response_indices.append(i)
            
            if not question_indices or not response_indices:
                return None
            
            reaction_times = []
            
            for q_idx in question_indices:
                q_time = onsets[q_idx]
                
                for r_idx in response_indices:
                    if r_idx <= q_idx:
                        continue
                    
                    r_time = onsets[r_idx]
                    rt = (r_time - q_time) * 1000
                    
                    if 200 < rt < 3000:
                        reaction_times.append(rt)
                        break
            
            if not reaction_times:
                return None
            
            return np.mean(reaction_times)
        
        except Exception as e:
            print(f"      RT extraction error: {e}")
            return None
    
    def extract_all_features_by_block(self):
        results = {
            'p300': {
                'honest_true': {}, 'deceitful_true': {},
                'honest_fake': {}, 'deceitful_fake': {}
            },
            'theta': {
                'honest_true': {}, 'deceitful_true': {},
                'honest_fake': {}, 'deceitful_fake': {}
            },
            'alpha': {
                'honest_true': {}, 'deceitful_true': {},
                'honest_fake': {}, 'deceitful_fake': {}
            },
            'beta': {
                'honest_true': {}, 'deceitful_true': {},
                'honest_fake': {}, 'deceitful_fake': {}
            },
            'gamma': {
                'honest_true': {}, 'deceitful_true': {},
                'honest_fake': {}, 'deceitful_fake': {}
            },
            'response_time': {
                'honest_true': {}, 'deceitful_true': {},
                'honest_fake': {}, 'deceitful_fake': {}
            }
        }
        
        print("\nExtracting features...")
        print("Using MNE signal processing...")
        print("Gamma extraction with 50Hz/100Hz notch filtering...\n")
        
        total_participants = len(self.all_data)
        for idx, (participant_id, blocks) in enumerate(self.all_data.items(), 1):
            print(f"  [{idx}/{total_participants}] Processing {participant_id}...")
            
            for block_key, raw in blocks.items():
                if raw is None or block_key not in results['p300']:
                    continue
                
                p300 = self.extract_p300_from_raw(raw)
                if p300 is not None:
                    results['p300'][block_key][participant_id] = float(p300)
                    print(f"    {block_key}: P300 = {p300:.2f} µV")
                
                theta = self.extract_frequency_power(raw, (4, 8))
                if theta is not None:
                    results['theta'][block_key][participant_id] = float(theta)
                
                alpha = self.extract_frequency_power(raw, (8, 13))
                if alpha is not None:
                    results['alpha'][block_key][participant_id] = float(alpha)
                
                beta = self.extract_frequency_power(raw, (13, 30))
                if beta is not None:
                    results['beta'][block_key][participant_id] = float(beta)
                
                gamma = self.extract_gamma_power(raw)
                if gamma is not None:
                    results['gamma'][block_key][participant_id] = float(gamma)
                
                rt = self.extract_rt_from_annotations(raw)
                if rt is not None:
                    results['response_time'][block_key][participant_id] = float(rt)
        
        return results


def save_features_to_pickle(results, filename='results/real_features.pkl'):
    Path('results').mkdir(exist_ok=True)
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nFeatures saved to {filename}")
    print("  Load in EDA notebook with:")
    print("  import pickle")
    print(f"  with open('{filename}', 'rb') as f:")
    print("      real_features = pickle.load(f)")


if __name__ == '__main__':
    extractor = RealFeatureExtractor()
    results = extractor.extract_all_features_by_block()
    save_features_to_pickle(results)
