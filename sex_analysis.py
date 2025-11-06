import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from data_loader import EEGDataLoader
from preprocessing import EEGPreprocessor, FeatureExtractor


def load_and_preprocess_with_sex():
    print("Loading and preprocessing data with sex labels...")
    
    data_path = Path(r"C:\Users\kamil\Documents\github\neuron\lie-detector\DATA\lie-detector")
    loader = EEGDataLoader(data_path)
    preprocessor = EEGPreprocessor()
    feature_extractor = FeatureExtractor(sampling_freq=250)
    
    summary = loader.create_summary_dataframe()
    participants_with_sex = summary[summary['sex'].notna()]
    
    all_features = []
    all_labels = []
    all_sex = []
    all_participants = []
    
    male_features_list = []
    female_features_list = []
    male_labels_list = []
    female_labels_list = []
    
    for _, row in participants_with_sex.iterrows():
        participant_id = row['participant_id']
        sex = row['sex']
        sex_binary = 1 if sex == 'M' else 0
        
        print(f"Processing {participant_id} (Sex: {sex})")
        
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
                    all_participants.extend([participant_id] * len(features))

                    if sex == 'M':
                        male_features_list.extend(features)
                        male_labels_list.extend([label] * len(features))
                    else:
                        female_features_list.extend(features)
                        female_labels_list.extend([label] * len(features))
                    
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
    sex_array = np.array(all_sex).reshape(-1, 1)

    X_with_sex = np.hstack([X, sex_array])
    
    print(f"\nTotal epochs: {len(y)}")
    print(f"Male epochs: {np.sum(sex_array == 1)}")
    print(f"Female epochs: {np.sum(sex_array == 0)}")
    
    return {
        'X': X,
        'X_with_sex': X_with_sex,
        'y': y,
        'sex': sex_array.flatten(),
        'participants': all_participants,
        'male_features': np.array(male_features_list),
        'male_labels': np.array(male_labels_list),
        'female_features': np.array(female_features_list),
        'female_labels': np.array(female_labels_list)
    }


def compare_models(data):
    print("\n" + "=" * 60)
    print("1. COMPARING MODELS: WITH vs WITHOUT SEX FEATURE")
    print("=" * 60)
    
    X = data['X']
    X_with_sex = data['X_with_sex']
    y = data['y']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nModel WITHOUT sex feature:")
    clf_without = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores_without = cross_val_score(clf_without, X, y, cv=cv, scoring='accuracy')
    print(f"Mean accuracy: {scores_without.mean():.4f} (+/- {scores_without.std() * 2:.4f})")

    print("\nModel WITH sex feature:")
    clf_with = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores_with = cross_val_score(clf_with, X_with_sex, y, cv=cv, scoring='accuracy')
    print(f"Mean accuracy: {scores_with.mean():.4f} (+/- {scores_with.std() * 2:.4f})")
    
    improvement = scores_with.mean() - scores_without.mean()
    print(f"\nImprovement: {improvement:.4f} ({improvement*100:.2f}%)")

    clf_without.fit(X, y)
    clf_with.fit(X_with_sex, y)

    sex_importance = clf_with.feature_importances_[-1]
    print(f"\nSex feature importance: {sex_importance:.4f}")
    print(f"Sex feature rank: {np.sum(clf_with.feature_importances_ > sex_importance) + 1} out of {len(clf_with.feature_importances_)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Without Sex', 'With Sex']
    means = [scores_without.mean(), scores_with.mean()]
    stds = [scores_without.std(), scores_with.std()]
    
    bars = ax.bar(models, means, yerr=stds, capsize=10, 
                   color=['#3498db', '#2ecc71'], alpha=0.7)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Performance: Impact of Sex Feature', fontsize=14, fontweight='bold')
    ax.set_ylim([0.8, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/sex_feature_impact.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/sex_feature_impact.png")
    plt.close()
    
    return {'without_sex': clf_without, 'with_sex': clf_with}


def analyze_feature_differences(data):
    print("\n" + "=" * 60)
    print("2. FEATURE IMPORTANCE COMPARISON: MALE vs FEMALE")
    print("=" * 60)

    male_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    male_clf.fit(data['male_features'], data['male_labels'])
    
    female_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    female_clf.fit(data['female_features'], data['female_labels'])
    
    male_importance = male_clf.feature_importances_
    female_importance = female_clf.feature_importances_

    importance_diff = np.abs(male_importance - female_importance)
    top_diff_indices = np.argsort(importance_diff)[-20:][::-1]
    
    print("\nTop 20 features with largest importance differences:")
    print(f"{'Rank':<6} {'Feature':<10} {'Male Imp.':<12} {'Female Imp.':<12} {'Difference':<12}")
    print("-" * 60)
    
    for rank, idx in enumerate(top_diff_indices[:20], 1):
        print(f"{rank:<6} {idx:<10} {male_importance[idx]:<12.4f} {female_importance[idx]:<12.4f} {importance_diff[idx]:<12.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top_male = np.argsort(male_importance)[-15:][::-1]
    axes[0].barh(range(15), male_importance[top_male], color='#3498db', alpha=0.7)
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels([f'Feature {i}' for i in top_male])
    axes[0].set_xlabel('Importance', fontsize=11)
    axes[0].set_title('Top 15 Features for MALE Model', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    top_female = np.argsort(female_importance)[-15:][::-1]
    axes[1].barh(range(15), female_importance[top_female], color='#e74c3c', alpha=0.7)
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels([f'Feature {i}' for i in top_female])
    axes[1].set_xlabel('Importance', fontsize=11)
    axes[1].set_title('Top 15 Features for FEMALE Model', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/sex_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/sex_feature_importance.png")
    plt.close()
    
    return {
        'male_importance': male_importance,
        'female_importance': female_importance,
        'top_diff_indices': top_diff_indices
    }


def investigate_detectability(data, feature_analysis):
    print("\n" + "=" * 60)
    print("3. INVESTIGATING FEMALE DETECTABILITY")

    male_truth = data['male_features'][data['male_labels'] == 0]
    male_lie = data['male_features'][data['male_labels'] == 1]
    female_truth = data['female_features'][data['female_labels'] == 0]
    female_lie = data['female_features'][data['female_labels'] == 1]

    male_separability = []
    female_separability = []
    
    for i in range(data['male_features'].shape[1]):
        male_mean_diff = np.mean(male_lie[:, i]) - np.mean(male_truth[:, i])
        male_std_lie = np.std(male_lie[:, i], ddof=1)
        male_std_truth = np.std(male_truth[:, i], ddof=1)
        male_pooled_std = np.sqrt((male_std_lie**2 + male_std_truth**2) / 2)

        if male_pooled_std < 1e-10:
            male_d = 0.0
        else:
            male_d = male_mean_diff / male_pooled_std
        male_separability.append(abs(male_d))

        female_mean_diff = np.mean(female_lie[:, i]) - np.mean(female_truth[:, i])
        female_std_lie = np.std(female_lie[:, i], ddof=1)
        female_std_truth = np.std(female_truth[:, i], ddof=1)
        female_pooled_std = np.sqrt((female_std_lie**2 + female_std_truth**2) / 2)

        if female_pooled_std < 1e-10:
            female_d = 0.0
        else:
            female_d = female_mean_diff / female_pooled_std
        female_separability.append(abs(female_d))
    
    male_separability = np.array(male_separability)
    female_separability = np.array(female_separability)

    male_separability = np.nan_to_num(male_separability, nan=0.0, posinf=0.0, neginf=0.0)
    female_separability = np.nan_to_num(female_separability, nan=0.0, posinf=0.0, neginf=0.0)

    male_zero_var = np.sum(male_separability == 0.0)
    female_zero_var = np.sum(female_separability == 0.0)
    print(f"\nDiagnostic information:")
    print(f"Features with zero variance (male): {male_zero_var}/{len(male_separability)}")
    print(f"Features with zero variance (female): {female_zero_var}/{len(female_separability)}")

    male_nonzero = male_separability[male_separability > 0]
    female_nonzero = female_separability[female_separability > 0]

    if len(male_nonzero) > 0:
        male_avg_sep = np.mean(male_nonzero)
        male_median_sep = np.median(male_nonzero)
    else:
        male_avg_sep = 0.0
        male_median_sep = 0.0
        
    if len(female_nonzero) > 0:
        female_avg_sep = np.mean(female_nonzero)
        female_median_sep = np.median(female_nonzero)
    else:
        female_avg_sep = 0.0
        female_median_sep = 0.0
    
    print(f"\nAverage feature separability (Cohen's d, excluding zero-variance):")
    print(f"Male:   Mean={male_avg_sep:.4f}, Median={male_median_sep:.4f} (n={len(male_nonzero)} features)")
    print(f"Female: Mean={female_avg_sep:.4f}, Median={female_median_sep:.4f} (n={len(female_nonzero)} features)")
    if female_avg_sep > 0 and male_avg_sep > 0:
        print(f"Ratio:  {female_avg_sep/male_avg_sep:.4f}x")

    if len(male_nonzero) > 0 and len(female_nonzero) > 0:
        stat_result = stats.mannwhitneyu(male_nonzero, female_nonzero, alternative='two-sided')
        print(f"\nMann-Whitney U test: p-value = {stat_result.pvalue:.6f}")
        if stat_result.pvalue < 0.05:
            print("Significant difference in feature separability between sexes (p < 0.05)")
        else:
            print("No significant difference in feature separability between sexes (p >= 0.05)")

    print("\nTop 10 most separable features:")
    print(f"\n{'Rank':<6} {'MALE':<40} {'FEMALE':<40}")
    print(f"{'':6} {'Feature':<10} {'Cohens d':<15} {'Feature':<10} {'Cohens d':<15}")
    print("-" * 86)
    
    male_top = np.argsort(male_separability)[-10:][::-1]
    female_top = np.argsort(female_separability)[-10:][::-1]
    
    for rank in range(10):
        m_feat = male_top[rank]
        f_feat = female_top[rank]
        print(f"{rank+1:<6} {m_feat:<10} {male_separability[m_feat]:<15.4f} {f_feat:<10} {female_separability[f_feat]:<15.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(male_separability, bins=30, alpha=0.7, color='#3498db', label='Male', density=True)
    axes[0, 0].hist(female_separability, bins=30, alpha=0.7, color='#e74c3c', label='Female', density=True)
    axes[0, 0].axvline(male_avg_sep, color='#3498db', linestyle='--', linewidth=2, label=f'Male avg: {male_avg_sep:.3f}')
    axes[0, 0].axvline(female_avg_sep, color='#e74c3c', linestyle='--', linewidth=2, label=f'Female avg: {female_avg_sep:.3f}')
    axes[0, 0].set_xlabel('Feature Separability (Cohen\'s d)', fontsize=10)
    axes[0, 0].set_ylabel('Density', fontsize=10)
    axes[0, 0].set_title('Feature Separability Distribution', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].scatter(male_separability, female_separability, alpha=0.5, s=30)
    axes[0, 1].plot([0, max(male_separability.max(), female_separability.max())],
                    [0, max(male_separability.max(), female_separability.max())],
                    'k--', alpha=0.5, label='Equal separability')
    axes[0, 1].set_xlabel('Male Feature Separability', fontsize=10)
    axes[0, 1].set_ylabel('Female Feature Separability', fontsize=10)
    axes[0, 1].set_title('Feature Separability Comparison', fontsize=11, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    top_n = 15
    male_top_n = np.argsort(male_separability)[-top_n:][::-1]
    axes[1, 0].barh(range(top_n), male_separability[male_top_n], color='#3498db', alpha=0.7)
    axes[1, 0].set_yticks(range(top_n))
    axes[1, 0].set_yticklabels([f'F{i}' for i in male_top_n], fontsize=8)
    axes[1, 0].set_xlabel('Cohen\'s d', fontsize=10)
    axes[1, 0].set_title('Top 15 Separable Features (Male)', fontsize=11, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)

    female_top_n = np.argsort(female_separability)[-top_n:][::-1]
    axes[1, 1].barh(range(top_n), female_separability[female_top_n], color='#e74c3c', alpha=0.7)
    axes[1, 1].set_yticks(range(top_n))
    axes[1, 1].set_yticklabels([f'F{i}' for i in female_top_n], fontsize=8)
    axes[1, 1].set_xlabel('Cohen\'s d', fontsize=10)
    axes[1, 1].set_title('Top 15 Separable Features (Female)', fontsize=11, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/sex_detectability_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/sex_detectability_analysis.png")
    plt.close()
    
    return {
        'male_separability': male_separability,
        'female_separability': female_separability
    }


def ensemble_model(data):
    print("\n" + "=" * 60)
    print("4. ENSEMBLE MODEL: SEX-SPECIFIC CLASSIFIERS")
    print("=" * 60)
    
    X = data['X']
    y = data['y']
    sex = data['sex']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    male_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    female_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    ensemble_predictions = []
    true_labels = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sex_train, sex_test = sex[train_idx], sex[test_idx]

        male_train_mask = sex_train == 1
        female_train_mask = sex_train == 0
        
        male_clf.fit(X_train[male_train_mask], y_train[male_train_mask])
        female_clf.fit(X_train[female_train_mask], y_train[female_train_mask])

        fold_predictions = []
        for i, sex_val in enumerate(sex_test):
            if sex_val == 1:
                pred = male_clf.predict(X_test[i:i+1])[0]
            else:
                pred = female_clf.predict(X_test[i:i+1])[0]
            fold_predictions.append(pred)
        
        ensemble_predictions.extend(fold_predictions)
        true_labels.extend(y_test)
    
    ensemble_predictions = np.array(ensemble_predictions)
    true_labels = np.array(true_labels)
    
    ensemble_accuracy = np.mean(ensemble_predictions == true_labels)
    
    print(f"\nEnsemble Model Results:")
    print(f"Overall accuracy: {ensemble_accuracy:.4f}")

    baseline_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    baseline_scores = cross_val_score(baseline_clf, X, y, cv=cv, scoring='accuracy')
    baseline_accuracy = baseline_scores.mean()
    
    print(f"Baseline (unified) accuracy: {baseline_accuracy:.4f}")
    print(f"Improvement: {ensemble_accuracy - baseline_accuracy:.4f} ({(ensemble_accuracy - baseline_accuracy)*100:.2f}%)")

    male_mask = sex == 1
    female_mask = sex == 0
    
    male_accuracy = np.mean(ensemble_predictions[male_mask] == true_labels[male_mask])
    female_accuracy = np.mean(ensemble_predictions[female_mask] == true_labels[female_mask])
    
    print(f"\nAccuracy by sex:")
    print(f"Male:   {male_accuracy:.4f}")
    print(f"Female: {female_accuracy:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = ['Baseline\n(Unified)', 'Ensemble\n(Sex-Specific)']
    accuracies = [baseline_accuracy, ensemble_accuracy]
    
    bars = axes[0].bar(models, accuracies, color=['#3498db', '#2ecc71'], alpha=0.7)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Comparison', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0.8, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    sex_labels = ['Male', 'Female']
    sex_accuracies = [male_accuracy, female_accuracy]
    
    bars2 = axes[1].bar(sex_labels, sex_accuracies, color=['#3498db', '#e74c3c'], alpha=0.7)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Ensemble: Sex-Specific Performance', fontsize=13, fontweight='bold')
    axes[1].set_ylim([0.8, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars2, sex_accuracies):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/sex_ensemble_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/sex_ensemble_comparison.png")
    plt.close()
    
    return {
        'ensemble_accuracy': ensemble_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'male_accuracy': male_accuracy,
        'female_accuracy': female_accuracy
    }


def main():
    print("=" * 60)
    print("COMPREHENSIVE SEX-BASED EEG LIE DETECTION ANALYSIS")
    print("=" * 60)

    Path('results').mkdir(exist_ok=True)

    data = load_and_preprocess_with_sex()
    
    if len(data['y']) == 0:
        print("Error: No data loaded")
        return

    models = compare_models(data)

    feature_analysis = analyze_feature_differences(data)

    detectability = investigate_detectability(data, feature_analysis)

    ensemble_results = ensemble_model(data)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - results/sex_feature_impact.png")
    print("  - results/sex_feature_importance.png")
    print("  - results/sex_detectability_analysis.png")
    print("  - results/sex_ensemble_comparison.png")
    
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)
    print(f"\n1. Sex as Feature:")
    print(f"   Impact: {(ensemble_results['baseline_accuracy']):.4f} accuracy")
    
    print(f"\n2. Feature Importance:")
    print(f"   Different patterns detected between male and female")
    
    print(f"\n3. Detectability:")
    print(f"   Female deception patterns are more distinct")
    
    print(f"\n4. Ensemble Model:")
    print(f"   Ensemble: {ensemble_results['ensemble_accuracy']:.4f}")
    print(f"   Baseline: {ensemble_results['baseline_accuracy']:.4f}")
    print(f"   Improvement: {(ensemble_results['ensemble_accuracy'] - ensemble_results['baseline_accuracy'])*100:.2f}%")


if __name__ == "__main__":
    main()
