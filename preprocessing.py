import numpy as np
import mne
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import skew, kurtosis


class EEGPreprocessor:
    def __init__(self, 
                 l_freq: float = 0.1,
                 h_freq: float = 30.0,
                 notch_freq: float = 50.0,
                 epoch_tmin: float = -0.2,
                 epoch_tmax: float = 0.8):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.epoch_tmin = epoch_tmin
        self.epoch_tmax = epoch_tmax
    
    def filter_raw(self, raw: mne.io.Raw, copy: bool = True) -> mne.io.Raw:
        if copy:
            raw = raw.copy()
        
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, picks='eeg', verbose=False)
        raw.notch_filter(freqs=self.notch_freq, picks='eeg', verbose=False)
        
        return raw
    
    def create_epochs(self, raw: mne.io.Raw, 
                     event_id: Optional[Dict] = None,
                     baseline: Optional[Tuple] = (-0.2, 0),
                     reject: Optional[Dict] = None) -> mne.Epochs:
        
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        
        stimulus_event_ids = {k: v for k, v in event_dict.items() 
                              if 'PersonalDataField' in k}
        
        if not stimulus_event_ids:
            print("    Warning: No stimulus events found")
            empty_events = np.empty((0, 3), dtype=int)
            return mne.Epochs(raw, empty_events, {}, 
                            tmin=self.epoch_tmin, tmax=self.epoch_tmax,
                            preload=True, verbose=False)
        
        event_id = event_id if event_id is not None else stimulus_event_ids
        
        epochs = mne.Epochs(raw, events, event_id, 
                           tmin=self.epoch_tmin, tmax=self.epoch_tmax,
                           baseline=baseline, reject=reject, 
                           preload=True, verbose=False)
        
        return epochs
    
    def preprocess_pipeline(self, raw: mne.io.Raw, 
                           event_id: Optional[Dict] = None) -> mne.Epochs:
        
        raw_filtered = self.filter_raw(raw, copy=True)
        epochs = self.create_epochs(raw_filtered, event_id=event_id)
        
        print(f"  Created {len(epochs)} epochs from {len(epochs.events)} events")
        
        return epochs


class FeatureExtractor:
    FREQ_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
    }
    
    def __init__(self, sampling_freq: float = 250.0):
        self.sfreq = sampling_freq
    
    def extract_statistical_features(self, epochs: mne.Epochs) -> np.ndarray:
        data = epochs.get_data()
        features = []
        
        for epoch in data:
            epoch_features = []
            for channel in epoch:
                epoch_features.extend([
                    np.mean(channel), np.std(channel), np.var(channel),
                    skew(channel), kurtosis(channel),
                    np.min(channel), np.max(channel), np.ptp(channel)
                ])
            features.append(epoch_features)
        
        return np.array(features)
    
    def extract_frequency_features(self, epochs: mne.Epochs) -> np.ndarray:
        data = epochs.get_data()
        features = []
        
        for epoch in data:
            epoch_features = []
            for channel in epoch:
                freqs, psd = signal.welch(channel, fs=self.sfreq, 
                                         nperseg=min(256, len(channel)))
                
                for band_name, (low, high) in self.FREQ_BANDS.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    band_power = np.mean(psd[band_mask])
                    epoch_features.append(band_power)
            
            features.append(epoch_features)
        
        return np.array(features)
    
    def extract_combined_features(self, epochs: mne.Epochs) -> np.ndarray:
        stat_features = self.extract_statistical_features(epochs)
        freq_features = self.extract_frequency_features(epochs)
        combined = np.hstack([stat_features, freq_features])
        
        print(f"  Extracted features: {combined.shape[1]} features per epoch")
        
        return combined
    
    def get_feature_names(self, channel_names: List[str]) -> List[str]:
        feature_names = []
        stat_names = ['mean', 'std', 'var', 'skew', 'kurt', 'min', 'max', 'ptp']
        
        for ch in channel_names:
            for stat in stat_names:
                feature_names.append(f"{ch}_{stat}")
        
        for ch in channel_names:
            for band in self.FREQ_BANDS.keys():
                feature_names.append(f"{ch}_{band}_power")
        
        return feature_names


def create_binary_labels(epochs: mne.Epochs, 
                        honest_events: List[str],
                        deceitful_events: List[str]) -> np.ndarray:
    labels = []
    
    for event_name in epochs.events[:, 2]:
        event_names = [name for name, id_ in epochs.event_id.items() if id_ == event_name]
        
        if event_names:
            event_str = event_names[0]
            if any(honest in event_str for honest in honest_events):
                labels.append(0)
            elif any(deceitful in event_str for deceitful in deceitful_events):
                labels.append(1)
            else:
                labels.append(-1)
    
    return np.array(labels)


if __name__ == "__main__":
    print("EEG Preprocessing Module")
