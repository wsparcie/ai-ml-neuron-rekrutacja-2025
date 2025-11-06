import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from data_loader import EEGDataLoader
from preprocessing import EEGPreprocessor, FeatureExtractor


def load_and_preprocess_with_demographics():
    print("Loading and preprocessing data with demographics...")
    
    data_path = Path(r"C:\Users\kamil\Documents\github\neuron\lie-detector\DATA\lie-detector")
    loader = EEGDataLoader(data_path)
    preprocessor = EEGPreprocessor()
    feature_extractor = FeatureExtractor(sampling_freq=250)
    
    summary = loader.create_summary_dataframe()
    participants_with_metadata = summary[(summary['sex'].notna()) & (summary['age'].notna())]
    
    print(f"\nParticipants with complete metadata: {len(participants_with_metadata)}")
    print(f"Age range: {participants_with_metadata['age'].min()}-{participants_with_metadata['age'].max()}")
    print(f"Sex distribution: {participants_with_metadata['sex'].value_counts().to_dict()}")
    
    all_features = []
    all_labels = []
    all_sex = []
    all_age = []
    all_participants = []
    
    for _, row in participants_with_metadata.iterrows():
        participant_id = row['participant_id']
        sex = row['sex']
        age = row['age']
        sex_binary = 1 if sex == 'M' else 0
        
        print(f"Processing {participant_id} (Sex: {sex}, Age: {age})")
        
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
                    
                    all_features.extend(features)
                    all_labels.extend([label] * len(features))
                    all_sex.extend([sex_binary] * len(features))
                    all_age.extend([age] * len(features))
                    all_participants.extend([participant_id] * len(features))
                    
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
    
    X = np.array(all_features)
    y = np.array(all_labels)
    sex_array = np.array(all_sex)
    age_array = np.array(all_age)
    
    print(f"\nTotal epochs: {len(y)}")
    print(f"Male epochs: {np.sum(sex_array == 1)}")
    print(f"Female epochs: {np.sum(sex_array == 0)}")
    
    return {
        'X': X,
        'y': y,
        'sex': sex_array,
        'age': age_array,
        'participants': all_participants,
        'metadata': participants_with_metadata
    }


def analyze_age_distribution(data):
    print("\n" + "=" * 60)
    print("AGE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    ages = data['age']
    sex = data['sex']
    
    male_ages = ages[sex == 1]
    female_ages = ages[sex == 0]
    
    print(f"\nOverall age statistics:")
    print(f"  Mean: {np.mean(ages):.1f} years")
    print(f"  Median: {np.median(ages):.1f} years")
    print(f"  Range: {np.min(ages)}-{np.max(ages)} years")
    print(f"  Std: {np.std(ages):.1f} years")
    
    print(f"\nAge by sex:")
    print(f"  Male:   Mean={np.mean(male_ages):.1f}, Range={np.min(male_ages)}-{np.max(male_ages)}")
    print(f"  Female: Mean={np.mean(female_ages):.1f}, Range={np.min(female_ages)}-{np.max(female_ages)}")
    
    stat_result = stats.mannwhitneyu(male_ages, female_ages, alternative='two-sided')
    print(f"\nMann-Whitney U test (age difference between sexes): p={stat_result.pvalue:.4f}")
    if stat_result.pvalue < 0.05:
        print("Significant age difference between sexes")
    else:
        print("No significant age difference between sexes")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(male_ages, bins=10, alpha=0.7, color='#3498db', label='Male', edgecolor='black')
    axes[0, 0].hist(female_ages, bins=10, alpha=0.7, color='#e74c3c', label='Female', edgecolor='black')
    axes[0, 0].axvline(np.mean(male_ages), color='#3498db', linestyle='--', linewidth=2, label=f'Male mean: {np.mean(male_ages):.1f}')
    axes[0, 0].axvline(np.mean(female_ages), color='#e74c3c', linestyle='--', linewidth=2, label=f'Female mean: {np.mean(female_ages):.1f}')
    axes[0, 0].set_xlabel('Age (years)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Age Distribution by Sex', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    box_data = [male_ages, female_ages]
    bp = axes[0, 1].boxplot(box_data, labels=['Male', 'Female'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    axes[0, 1].set_ylabel('Age (years)', fontsize=11)
    axes[0, 1].set_title('Age Distribution Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')

    unique_ages, counts = np.unique(ages, return_counts=True)
    axes[1, 0].bar(unique_ages, counts, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Age (years)', fontsize=11)
    axes[1, 0].set_ylabel('Number of Epochs', fontsize=11)
    axes[1, 0].set_title('Sample Distribution by Age', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')

    male_mask = sex == 1
    female_mask = sex == 0
    axes[1, 1].scatter(ages[male_mask], np.random.normal(1, 0.05, np.sum(male_mask)), 
                       alpha=0.5, s=50, color='#3498db', label='Male')
    axes[1, 1].scatter(ages[female_mask], np.random.normal(0, 0.05, np.sum(female_mask)), 
                       alpha=0.5, s=50, color='#e74c3c', label='Female')
    axes[1, 1].set_xlabel('Age (years)', fontsize=11)
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(['Female', 'Male'])
    axes[1, 1].set_title('Age-Sex Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/age_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/age_distribution_analysis.png")
    plt.close()


def test_demographic_features(data):
    print("\n" + "=" * 60)
    print("DEMOGRAPHIC FEATURES IMPACT")
    print("=" * 60)
    
    X = data['X']
    y = data['y']
    sex = data['sex'].reshape(-1, 1)
    age = data['age'].reshape(-1, 1)

    age_normalized = (age - np.mean(age)) / np.std(age)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}

    print("\n1. Baseline (no demographics):")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    results['baseline'] = scores.mean()
    print(f"   Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    print("\n2. With SEX feature:")
    X_sex = np.hstack([X, sex])
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X_sex, y, cv=cv, scoring='accuracy')
    results['with_sex'] = scores.mean()
    improvement_sex = scores.mean() - results['baseline']
    print(f"   Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"   Improvement: {improvement_sex:+.4f} ({improvement_sex*100:+.2f}%)")

    print("\n3. With AGE feature:")
    X_age = np.hstack([X, age_normalized])
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X_age, y, cv=cv, scoring='accuracy')
    results['with_age'] = scores.mean()
    improvement_age = scores.mean() - results['baseline']
    print(f"   Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"   Improvement: {improvement_age:+.4f} ({improvement_age*100:+.2f}%)")

    print("\n4. With BOTH sex and age:")
    X_both = np.hstack([X, sex, age_normalized])
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X_both, y, cv=cv, scoring='accuracy')
    results['with_both'] = scores.mean()
    improvement_both = scores.mean() - results['baseline']
    print(f"   Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"   Improvement: {improvement_both:+.4f} ({improvement_both*100:+.2f}%)")

    clf.fit(X_both, y)
    sex_importance = clf.feature_importances_[-2]
    age_importance = clf.feature_importances_[-1]
    print(f"\n5. Feature importance:")
    print(f"   Sex feature: {sex_importance:.6f} (rank {np.sum(clf.feature_importances_ > sex_importance) + 1}/254)")
    print(f"   Age feature: {age_importance:.6f} (rank {np.sum(clf.feature_importances_ > age_importance) + 1}/254)")

    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = ['Baseline\n(No Demographics)', 'With SEX', 'With AGE', 'With BOTH']
    accuracies = [results['baseline'], results['with_sex'], results['with_age'], results['with_both']]
    colors = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71']
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Impact of Demographic Features on Model Performance', fontsize=14, fontweight='bold')
    ax.set_ylim([0.8, 1.0])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=results['baseline'], color='gray', linestyle='--', alpha=0.5, label='Baseline')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        improvement = acc - results['baseline']
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}\n({improvement:+.3f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/demographic_features_impact.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/demographic_features_impact.png")
    plt.close()
    
    return results


def analyze_age_groups(data):
    print("\n" + "=" * 60)
    print("AGE GROUP ANALYSIS")
    print("=" * 60)

    ages = data['age']
    age_bins = [18, 21, 23, 27]
    age_labels = ['18-20', '21-22', '23-26']
    age_groups = pd.cut(ages, bins=age_bins, labels=age_labels, include_lowest=True)
    
    X = data['X']
    y = data['y']
    sex = data['sex']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    for age_label in age_labels:
        mask = age_groups == age_label
        if np.sum(mask) < 50:
            print(f"\n{age_label}: Too few samples ({np.sum(mask)}), skipping")
            continue
        
        X_age = X[mask]
        y_age = y[mask]
        sex_age = sex[mask]
        
        n_male = np.sum(sex_age == 1)
        n_female = np.sum(sex_age == 0)
        
        print(f"\n{age_label} years:")
        print(f"  Total epochs: {len(y_age)}")
        print(f"  Male: {n_male}, Female: {n_female}")

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        if len(np.unique(y_age)) > 1:
            scores = cross_val_score(clf, X_age, y_age, cv=min(cv.n_splits, len(y_age)//2), scoring='accuracy')
            mean_acc = scores.mean()
            std_acc = scores.std()
            
            print(f"  Accuracy: {mean_acc:.4f} (+/- {std_acc * 2:.4f})")
            
            results.append({
                'age_group': age_label,
                'accuracy': mean_acc,
                'std': std_acc,
                'n_samples': len(y_age),
                'n_male': n_male,
                'n_female': n_female
            })
        else:
            print(f"  Only one class present, skipping")
    
    if len(results) > 0:
        results_df = pd.DataFrame(results)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].bar(results_df['age_group'], results_df['accuracy'], 
                    yerr=results_df['std'], capsize=10,
                    color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[0].set_xlabel('Age Group (years)', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Lie Detection Accuracy by Age Group', fontsize=13, fontweight='bold')
        axes[0].set_ylim([0.7, 1.0])
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, row in results_df.iterrows():
            axes[0].text(i, row['accuracy'] + row['std'] + 0.02,
                        f"{row['accuracy']:.3f}\nn={row['n_samples']}",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        x = np.arange(len(results_df))
        width = 0.35
        axes[1].bar(x - width/2, results_df['n_male'], width, label='Male', 
                    color='#3498db', alpha=0.7, edgecolor='black')
        axes[1].bar(x + width/2, results_df['n_female'], width, label='Female', 
                    color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Age Group (years)', fontsize=12)
        axes[1].set_ylabel('Number of Epochs', fontsize=12)
        axes[1].set_title('Sample Distribution by Age and Sex', fontsize=13, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(results_df['age_group'])
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/age_group_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSaved: results/age_group_analysis.png")
        plt.close()
        
        return results_df
    else:
        print("\nNot enough data for age group analysis")
        return None


def analyze_age_sex_interaction(data):
    print("\n" + "=" * 60)
    print("AGE-SEX INTERACTION ANALYSIS")
    print("=" * 60)
    
    X = data['X']
    y = data['y']
    sex = data['sex']
    age = data['age']

    male_mask = sex == 1
    female_mask = sex == 0

    print("\nAge-accuracy correlation by sex:")
    
    for sex_name, mask in [('Male', male_mask), ('Female', female_mask)]:
        ages_subset = age[mask]
        X_subset = X[mask]
        y_subset = y[mask]
        
        if len(np.unique(ages_subset)) < 2:
            print(f"\n{sex_name}: Not enough age variation")
            continue

        unique_ages = np.unique(ages_subset)
        age_accuracies = []
        
        for age_val in unique_ages:
            age_mask = ages_subset == age_val
            X_age = X_subset[age_mask]
            y_age = y_subset[age_mask]
            
            if len(y_age) > 20 and len(np.unique(y_age)) > 1:
                clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                cv = StratifiedKFold(n_splits=min(3, len(y_age)//10), shuffle=True, random_state=42)
                scores = cross_val_score(clf, X_age, y_age, cv=cv, scoring='accuracy')
                age_accuracies.append((age_val, scores.mean(), len(y_age)))
        
        if len(age_accuracies) >= 2:
            age_vals = [x[0] for x in age_accuracies]
            accs = [x[1] for x in age_accuracies]
            
            corr, p_value = stats.pearsonr(age_vals, accs)
            print(f"\n{sex_name}:")
            print(f"  Correlation (age vs accuracy): r={corr:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                print(f"  Significant correlation!")
            else:
                print(f"  No significant correlation")

            print(f"  Per-age accuracy:")
            for age_val, acc, n in age_accuracies:
                print(f"    Age {age_val}: {acc:.3f} (n={n})")
        else:
            print(f"\n{sex_name}: Not enough age groups for correlation")
    
    print("\n" + "=" * 60)


def main():
    print("=" * 60)
    print("COMPREHENSIVE AGE & SEX ANALYSIS")
    print("=" * 60)
    
    Path('results').mkdir(exist_ok=True)

    data = load_and_preprocess_with_demographics()
    
    if len(data['y']) == 0:
        print("Error: No data loaded")
        return

    analyze_age_distribution(data)

    demo_results = test_demographic_features(data)

    age_group_results = analyze_age_groups(data)

    analyze_age_sex_interaction(data)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - results/age_distribution_analysis.png")
    print("  - results/demographic_features_impact.png")
    print("  - results/age_group_analysis.png")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n1. Demographic Features Impact:")
    print(f"   Baseline: {demo_results['baseline']:.4f}")
    print(f"   With SEX: {demo_results['with_sex']:.4f} ({(demo_results['with_sex']-demo_results['baseline'])*100:+.2f}%)")
    print(f"   With AGE: {demo_results['with_age']:.4f} ({(demo_results['with_age']-demo_results['baseline'])*100:+.2f}%)")
    print(f"   With BOTH: {demo_results['with_both']:.4f} ({(demo_results['with_both']-demo_results['baseline'])*100:+.2f}%)")
    
    if age_group_results is not None:
        print(f"\n2. Age Group Performance:")
        for _, row in age_group_results.iterrows():
            print(f"   {row['age_group']}: {row['accuracy']:.4f} (n={row['n_samples']})")


if __name__ == "__main__":
    main()
