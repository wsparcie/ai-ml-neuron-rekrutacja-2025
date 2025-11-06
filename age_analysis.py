import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from data_loader import EEGDataLoader
from preprocessing import EEGPreprocessor, FeatureExtractor


def load_and_preprocess_by_age():
    print("Loading and preprocessing data by age...")
    
    data_path = Path(r"C:\Users\kamil\Documents\github\neuron\lie-detector\DATA\lie-detector")
    loader = EEGDataLoader(data_path)
    preprocessor = EEGPreprocessor()
    feature_extractor = FeatureExtractor(sampling_freq=250)
    
    summary = loader.create_summary_dataframe()
    participants_with_age = summary[summary['age'].notna()]
    
    print(f"\nParticipants with age metadata: {len(participants_with_age)}")
    print(f"Age range: {participants_with_age['age'].min()}-{participants_with_age['age'].max()}")
    print(f"Age distribution:\n{participants_with_age['age'].value_counts().sort_index()}")
    
    age_bins = [18, 21, 23, 27]
    age_labels = ['Young (18-20)', 'Middle (21-22)', 'Older (23-26)']
    participants_with_age['age_group'] = pd.cut(participants_with_age['age'], 
                                                  bins=age_bins, 
                                                  labels=age_labels, 
                                                  include_lowest=True)
    
    age_data = {label: {'features': [], 'labels': [], 'participants': []} 
                for label in age_labels}
    
    for _, row in participants_with_age.iterrows():
        participant_id = row['participant_id']
        age = row['age']
        age_group = row['age_group']
        
        print(f"Processing {participant_id} (Age: {age}, Group: {age_group})")
        
        try:
            participant_data = loader.load_participant_data(participant_id, preload=True)
            
            for block_name, raw in participant_data.items():
                if 'deceitful' in block_name.lower():
                    label = 1
                else:
                    label = 0
                
                epochs = preprocessor.preprocess_pipeline(raw)
                
                if epochs is None or len(epochs) == 0:
                    continue
                
                try:
                    features = feature_extractor.extract_combined_features(epochs)
                    
                    age_data[age_group]['features'].extend(features)
                    age_data[age_group]['labels'].extend([label] * len(features))
                    age_data[age_group]['participants'].extend([participant_id] * len(features))
                    
                    print(f"  {block_name}: {len(features)} epochs")
                    
                except KeyboardInterrupt:
                    raise
                except Exception as feat_err:
                    print(f"  Warning: Could not extract features from {block_name}: {feat_err}")
                    continue
                    
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  Error processing {participant_id}: {e}")
    
    for age_group in age_labels:
        age_data[age_group]['features'] = np.array(age_data[age_group]['features'])
        age_data[age_group]['labels'] = np.array(age_data[age_group]['labels'])
    
    print("\n" + "=" * 60)
    print("AGE GROUP SUMMARY")
    print("=" * 60)
    for age_group in age_labels:
        n_epochs = len(age_data[age_group]['labels'])
        if n_epochs > 0:
            n_truth = np.sum(age_data[age_group]['labels'] == 0)
            n_lie = np.sum(age_data[age_group]['labels'] == 1)
            print(f"{age_group}: {n_epochs} epochs ({n_truth} truth, {n_lie} lies)")
    
    return age_data, participants_with_age


def train_and_evaluate_age_models(age_data):
    print("\n" + "=" * 60)
    print("TRAINING AGE-SPECIFIC MODELS")
    print("=" * 60)
    
    results = {}
    
    for age_group in age_data.keys():
        X = age_data[age_group]['features']
        y = age_data[age_group]['labels']
        
        if len(X) < 50:
            print(f"\n{age_group.upper()}: Too few samples ({len(X)}), skipping")
            continue
        
        if len(np.unique(y)) < 2:
            print(f"\n{age_group.upper()}: Only one class, skipping")
            continue
        
        print(f"\n{age_group.upper()} MODEL:")
        print(f"Total epochs: {len(y)}")
        print(f"Truth epochs: {np.sum(y == 0)}")
        print(f"Lie epochs: {np.sum(y == 1)}")
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        n_splits = min(5, len(y) // 20)
        if n_splits < 2:
            n_splits = 2
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        clf.fit(X, y)
        
        results[age_group] = {
            'model': clf,
            'cv_scores': scores,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'n_samples': len(y)
        }
    
    return results


def compare_age_accuracies(results):
    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON BY AGE")
    print("=" * 60)
    
    if len(results) == 0:
        print("No results to compare")
        return
    
    for age_group, res in results.items():
        print(f"{age_group}: {res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f} (n={res['n_samples']})")
    
    if len(results) >= 2:
        age_groups = list(results.keys())
        accuracies = [results[ag]['cv_scores'] for ag in age_groups]
        
        if len(accuracies) > 2:
            stat, p_value = stats.kruskal(*accuracies)
            print(f"\nKruskal-Wallis test: H={stat:.4f}, p={p_value:.4f}")
            if p_value < 0.05:
                print("Significant difference in accuracy between age groups (p < 0.05)")
            else:
                print("No significant difference in accuracy between age groups (p >= 0.05)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    age_groups = list(results.keys())
    accuracies = [results[ag]['mean_accuracy'] for ag in age_groups]
    stds = [results[ag]['std_accuracy'] for ag in age_groups]
    colors = ['#3498db', '#e67e22', '#2ecc71'][:len(age_groups)]
    
    bars = axes[0].bar(age_groups, accuracies, yerr=stds, capsize=10, 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xlabel('Age Group', fontsize=12)
    axes[0].set_title('Lie Detection Accuracy by Age Group', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0.5, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)
    
    for bar, acc, std in zip(bars, accuracies, stds):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{acc:.1%}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    n_samples = [results[ag]['n_samples'] for ag in age_groups]
    
    bars2 = axes[1].bar(age_groups, n_samples, 
                        color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    axes[1].set_ylabel('Number of Epochs', fontsize=12)
    axes[1].set_xlabel('Age Group', fontsize=12)
    axes[1].set_title('Sample Size by Age Group', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=15)
    
    for bar, n in zip(bars2, n_samples):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'{n}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/age_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/age_accuracy_comparison.png")
    plt.close()


def analyze_age_distribution(participants_df):
    print("\n" + "=" * 60)
    print("AGE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    ages = participants_df['age'].dropna()
    
    print(f"\nAge statistics:")
    print(f"  Mean: {ages.mean():.1f} years")
    print(f"  Median: {ages.median():.0f} years")
    print(f"  Range: {ages.min()}-{ages.max()} years")
    print(f"  Std: {ages.std():.1f} years")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(ages, bins=10, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(ages.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ages.mean():.1f}')
    axes[0, 0].axvline(ages.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {ages.median():.0f}')
    axes[0, 0].set_xlabel('Age (years)', fontsize=11)
    axes[0, 0].set_ylabel('Number of Participants', fontsize=11)
    axes[0, 0].set_title('Age Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    bp = axes[0, 1].boxplot([ages], labels=['All Participants'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#9b59b6')
    bp['boxes'][0].set_alpha(0.7)
    axes[0, 1].set_ylabel('Age (years)', fontsize=11)
    axes[0, 1].set_title('Age Distribution Box Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    age_group_counts = participants_df['age_group'].value_counts().sort_index()
    axes[1, 0].bar(range(len(age_group_counts)), age_group_counts.values, 
                   color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(range(len(age_group_counts)))
    axes[1, 0].set_xticklabels(age_group_counts.index, rotation=15)
    axes[1, 0].set_ylabel('Number of Participants', fontsize=11)
    axes[1, 0].set_title('Participants by Age Group', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    for i, v in enumerate(age_group_counts.values):
        axes[1, 0].text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    age_counts = ages.value_counts().sort_index()
    axes[1, 1].bar(age_counts.index, age_counts.values, 
                   color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Age (years)', fontsize=11)
    axes[1, 1].set_ylabel('Number of Participants', fontsize=11)
    axes[1, 1].set_title('Participant Count by Age', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/age_distribution.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/age_distribution.png")
    plt.close()


def analyze_class_balance_by_age(age_data):
    print("\n" + "=" * 60)
    print("CLASS BALANCE BY AGE GROUP")
    print("=" * 60)
    
    valid_groups = {ag: data for ag, data in age_data.items() if len(data['labels']) > 0}
    
    if len(valid_groups) == 0:
        print("No data available")
        return
    
    fig, axes = plt.subplots(1, len(valid_groups), figsize=(6*len(valid_groups), 5))
    if len(valid_groups) == 1:
        axes = [axes]
    
    for idx, (age_group, ax) in enumerate(zip(valid_groups.keys(), axes)):
        labels = age_data[age_group]['labels']
        
        truth_count = np.sum(labels == 0)
        lie_count = np.sum(labels == 1)
        total = len(labels)
        
        print(f"\n{age_group}:")
        print(f"  Truth epochs: {truth_count} ({truth_count/total*100:.1f}%)")
        print(f"  Lie epochs:   {lie_count} ({lie_count/total*100:.1f}%)")
        
        colors = ['#2ecc71', '#e74c3c']
        ax.pie([truth_count, lie_count], labels=['Truth', 'Lie'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title(f'{age_group}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/age_class_balance.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/age_class_balance.png")
    plt.close()


def main():
    print("=" * 60)
    print("AGE-BASED EEG LIE DETECTION ANALYSIS")
    print("=" * 60)
    
    Path('results').mkdir(exist_ok=True)
    
    age_data, participants_df = load_and_preprocess_by_age()
    
    has_data = any(len(age_data[ag]['features']) > 0 for ag in age_data.keys())
    if not has_data:
        print("Error: No data loaded for any age group")
        return
    
    analyze_age_distribution(participants_df)
    
    analyze_class_balance_by_age(age_data)
    
    results = train_and_evaluate_age_models(age_data)
    
    if len(results) > 0:
        compare_age_accuracies(results)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - results/age_distribution.png")
    print("  - results/age_class_balance.png")
    if len(results) > 0:
        print("  - results/age_accuracy_comparison.png")


if __name__ == "__main__":
    main()
