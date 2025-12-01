import pandas as pd

def normalize_metadata_columns(metadata):
    column_mapping = {
        'Płeć': 'Sex',
        'Wiek': 'Age',
        'UUID': 'ID'
    }
    for polish_name, english_name in column_mapping.items():
        if polish_name in metadata.columns:
            metadata = metadata.rename(columns={polish_name: english_name})
    return metadata


def collect_feature_data(loader, metadata, real_features, feature_name):
    data = []
    for participant_id in loader.participants:
        participant_meta = metadata[metadata['ID'] == participant_id]
        if participant_meta.empty:
            continue
        
        sex = participant_meta['Sex'].iloc[0]
        
        for block_type in ['honest_true', 'deceitful_true', 'honest_fake', 'deceitful_fake']:
            if block_type not in real_features[feature_name]:
                continue
            if participant_id not in real_features[feature_name][block_type]:
                continue
            
            value = real_features[feature_name][block_type][participant_id]
            if value is None:
                continue
            
            record = {
                'participant': participant_id,
                'sex': sex,
                'block': block_type,
                feature_name: value
            }
            
            if feature_name in ['p300', 'response_time']:
                record['label'] = 'deceitful' if 'deceitful' in block_type else 'honest'
            
            data.append(record)
    
    return pd.DataFrame(data)


def collect_age_rt_data(loader, metadata, real_features):
    data = []
    for participant_id in loader.participants:
        participant_meta = metadata[metadata['ID'] == participant_id]
        if participant_meta.empty:
            continue
        
        age = participant_meta['Age'].iloc[0]
        if pd.isna(age):
            continue
        
        for block_type in ['honest_true', 'deceitful_true', 'honest_fake', 'deceitful_fake']:
            if block_type not in real_features['response_time']:
                continue
            if participant_id not in real_features['response_time'][block_type]:
                continue
            
            rt_val = real_features['response_time'][block_type][participant_id]
            if rt_val is None:
                continue
            
            data.append({
                'participant': participant_id,
                'age': age,
                'block': block_type,
                'response_time': rt_val
            })
    
    return pd.DataFrame(data)
