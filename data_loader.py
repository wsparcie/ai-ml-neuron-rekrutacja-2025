import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import mne
import numpy as np
import pandas as pd


class EEGDataLoader:
    BLOCK_TYPES = {
        'HONEST_RESPONSE_TO_TRUE_IDENTITY': 'honest_true',
        'DECEITFUL_RESPONSE_TO_TRUE_IDENTITY': 'deceitful_true',
        'HONEST_RESPONSE_TO_FAKE_IDENTITY': 'honest_fake',
        'DECEITFUL_RESPONSE_TO_FAKE_IDENTITY': 'deceitful_fake'
    }
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {data_root}")
        
        self.participants = self._scan_participants()
        self.metadata = self._load_metadata()
        print(f"Found {len(self.participants)} participants")
        if self.metadata is not None:
            print(f"Loaded metadata for {len(self.metadata)} participants")
    
    def _scan_participants(self) -> List[str]:
        excluded_folders = [
            'Popsute dane - ASD793JD',
            '6A517891 - stare dane'
        ]
        
        participants = []
        for item in self.data_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if item.name in excluded_folders:
                    print(f"Skipping excluded folder: {item.name}")
                    continue
                    
                fif_files = list(item.glob('*.fif'))
                if len(fif_files) > 0:
                    participants.append(item.name)
        return sorted(participants)
    
    def _load_metadata(self) -> Optional[pd.DataFrame]:
        metadata_file = self.data_root / 'Ankiety.csv'
        if not metadata_file.exists():
            print("Warning: Ankiety.csv not found")
            return None
        
        try:
            for encoding in ['utf-8', 'cp1250', 'iso-8859-2', 'windows-1250']:
                try:
                    df = pd.read_csv(metadata_file, sep=';', encoding=encoding)
                    df.columns = [col.strip() for col in df.columns]
                    df['UUID'] = df['UUID'].str.upper().str.strip()
                    print(f"Successfully loaded metadata with encoding: {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            
            print("Warning: Could not decode metadata file with any encoding")
            return None
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            return None
    
    def get_participant_sex(self, participant_id: str) -> Optional[str]:
        if self.metadata is None:
            return None
        
        participant_id_upper = participant_id.upper()
        match = self.metadata[self.metadata['UUID'] == participant_id_upper]
        
        if len(match) == 0:
            return None
        
        sex_col = [col for col in self.metadata.columns if 'e' in col.lower() and len(col) <= 5]
        if sex_col:
            return match[sex_col[0]].values[0]
        return None
    
    def get_participant_age(self, participant_id: str) -> Optional[int]:
        if self.metadata is None:
            return None
        
        participant_id_upper = participant_id.upper()
        match = self.metadata[self.metadata['UUID'] == participant_id_upper]
        
        if len(match) == 0:
            return None
        
        age_col = [col for col in self.metadata.columns if 'wiek' in col.lower() or 'age' in col.lower()]
        if age_col:
            age_value = match[age_col[0]].values[0]
            try:
                return int(age_value)
            except (ValueError, TypeError):
                return None
        return None
    
    def load_participant_data(self, participant_id: str, 
                            preload: bool = True) -> Dict[str, mne.io.Raw]:
        participant_path = self.data_root / participant_id
        if not participant_path.exists():
            raise ValueError(f"Participant not found: {participant_id}")
        
        data = {}
        for full_name, short_name in self.BLOCK_TYPES.items():
            file_pattern = f"*{full_name}_raw.fif"
            files = list(participant_path.glob(file_pattern))
            
            if len(files) == 1:
                try:
                    raw = mne.io.read_raw_fif(files[0], preload=preload, verbose=False)
                    data[short_name] = raw
                    print(f"  Loaded {short_name}: {raw.n_times} samples, "
                          f"{len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")
                except Exception as e:
                    print(f"  Error loading {short_name}: {e}")
            elif len(files) == 0:
                print(f"  Missing {short_name}")
            else:
                print(f"  Multiple files found for {short_name}, using first")
                raw = mne.io.read_raw_fif(files[0], preload=preload, verbose=False)
                data[short_name] = raw
        
        return data
    
    def get_events_from_raw(self, raw: mne.io.Raw) -> Tuple[np.ndarray, Dict]:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        return events, event_id
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        summary = []
        
        for participant in self.participants:
            participant_path = self.data_root / participant
            fif_files = list(participant_path.glob('*.fif'))
            
            row = {
                'participant_id': participant,
                'n_files': len(fif_files),
                'sex': self.get_participant_sex(participant),
                'age': self.get_participant_age(participant),
            }
            
            for full_name, short_name in self.BLOCK_TYPES.items():
                file_pattern = f"*{full_name}_raw.fif"
                files = list(participant_path.glob(file_pattern))
                row[short_name] = len(files) > 0
            
            summary.append(row)
        
        return pd.DataFrame(summary)
    
    def load_all_participants(self, max_participants: int = None) -> Dict:
        participants_to_load = self.participants[:max_participants] if max_participants else self.participants
        
        all_data = {}
        for i, participant in enumerate(participants_to_load, 1):
            print(f"\n[{i}/{len(participants_to_load)}] Loading {participant}...")
            try:
                data = self.load_participant_data(participant, preload=True)
                all_data[participant] = data
            except Exception as e:
                print(f"  Failed to load {participant}: {e}")
        
        return all_data


def get_default_data_path() -> Path:
    implementation_dir = Path(__file__).parent
    root_dir = implementation_dir.parent
    data_path = root_dir / 'DATA' / 'lie-detector'
    
    if data_path.exists():
        return data_path
    
    raise FileNotFoundError(
        "Does the data directory exist?\n"
        f"Searched: {data_path}"
    )


if __name__ == "__main__":
    print("EEG Data Loader Test\n" + "="*50)
    
    try:
        data_path = get_default_data_path()
        print(f"Data path: {data_path}\n")
        
        loader = EEGDataLoader(data_path)
        
        summary = loader.create_summary_dataframe()
        print("\nParticipant Summary:")
        print(summary.to_string())
        
        if len(loader.participants) > 0:
            print(f"\n\nLoading example participant: {loader.participants[0]}")
            data = loader.load_participant_data(loader.participants[0])
            
            if data:
                first_block = list(data.keys())[0]
                events, event_id = loader.get_events_from_raw(data[first_block])
                print(f"\nEvents in {first_block}:")
                print(f"  Total events: {len(events)}")
                print(f"  Event types: {event_id}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nDoes the data directory exist?.")
