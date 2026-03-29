import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mne
import pandas as pd

from preprocessing import EEGPreprocessor, FeatureExtractor
from data_loader import EEGDataLoader
from analysis_utils import normalize_metadata_columns, collect_feature_data

mne.set_log_level('ERROR')

EEG_CHANNEL_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2']
N_CHANNELS = len(EEG_CHANNEL_NAMES)
SFREQ = 250.0
N_EPOCHS = 10
N_TIMES = int(SFREQ * 1.0)   # 1 s epochs


def _make_epochs(n_epochs: int = N_EPOCHS, rng_seed: int = 0) -> mne.EpochsArray:
    rng = np.random.default_rng(rng_seed)
    info = mne.create_info(EEG_CHANNEL_NAMES, sfreq=SFREQ, ch_types='eeg')
    data = rng.standard_normal((n_epochs, N_CHANNELS, N_TIMES)).astype(np.float32) * 1e-6
    return mne.EpochsArray(data, info, verbose=False)


def test_statistical_feature_shape():
    epochs = _make_epochs()
    extractor = FeatureExtractor(sampling_freq=SFREQ)

    stat = extractor.extract_statistical_features(epochs)

    expected_cols = N_CHANNELS * 8   # 8 stats per channel
    assert stat.shape == (N_EPOCHS, expected_cols), (
        f"expected ({N_EPOCHS}, {expected_cols}), got {stat.shape}"
    )
    print(f"statistical features shape correct: {stat.shape}")


def test_frequency_feature_shape():
    epochs = _make_epochs()
    extractor = FeatureExtractor(sampling_freq=SFREQ)

    freq = extractor.extract_frequency_features(epochs)

    n_bands = len(FeatureExtractor.FREQ_BANDS)
    expected_cols = N_CHANNELS * n_bands
    assert freq.shape == (N_EPOCHS, expected_cols), (
        f"expected ({N_EPOCHS}, {expected_cols}), got {freq.shape}"
    )
    print(f"frequency features shape correct: {freq.shape}")


def test_combined_feature_shape():
    epochs = _make_epochs()
    extractor = FeatureExtractor(sampling_freq=SFREQ)

    combined = extractor.extract_combined_features(epochs)

    stat_cols = N_CHANNELS * 8
    freq_cols = N_CHANNELS * len(FeatureExtractor.FREQ_BANDS)
    expected_cols = stat_cols + freq_cols
    assert combined.shape == (N_EPOCHS, expected_cols), (
        f"expected ({N_EPOCHS}, {expected_cols}), got {combined.shape}"
    )
    print(f"combined features shape correct: {combined.shape}")


def test_no_nans_in_features():
    epochs = _make_epochs(rng_seed=1)
    extractor = FeatureExtractor(sampling_freq=SFREQ)

    combined = extractor.extract_combined_features(epochs)
    nan_count = int(np.isnan(combined).sum())
    assert nan_count == 0, f"found {nan_count} NaN values in extracted features"
    print("no NaN values in extracted features")


def test_frequency_features_nonnegative():
    epochs = _make_epochs(rng_seed=2)
    extractor = FeatureExtractor(sampling_freq=SFREQ)

    freq = extractor.extract_frequency_features(epochs)
    assert (freq >= 0).all(), "frequency features contain negative values"
    print("all frequency features are non-negative")


def test_feature_names_count_matches_features():
    epochs = _make_epochs()
    extractor = FeatureExtractor(sampling_freq=SFREQ)

    combined = extractor.extract_combined_features(epochs)
    names = extractor.get_feature_names(EEG_CHANNEL_NAMES)

    assert len(names) == combined.shape[1], (
        f"feature name count ({len(names)}) != feature vector length ({combined.shape[1]})"
    )
    print(f"feature name count matches: {len(names)}")


def test_features_vary_across_epochs():
    rng = np.random.default_rng(3)
    info = mne.create_info(EEG_CHANNEL_NAMES, sfreq=SFREQ, ch_types='eeg')
    d1 = rng.standard_normal((1, N_CHANNELS, N_TIMES)).astype(np.float32) * 1e-6
    d2 = rng.standard_normal((1, N_CHANNELS, N_TIMES)).astype(np.float32) * 1e-6

    e1 = mne.EpochsArray(d1, info, verbose=False)
    e2 = mne.EpochsArray(d2, info, verbose=False)

    extractor = FeatureExtractor(sampling_freq=SFREQ)
    f1 = extractor.extract_combined_features(e1)[0]
    f2 = extractor.extract_combined_features(e2)[0]

    assert not np.allclose(f1, f2), "different epochs produced identical feature vectors"
    print("features vary across different epochs")


def test_band_names_present_in_feature_names():
    extractor = FeatureExtractor(sampling_freq=SFREQ)
    names = extractor.get_feature_names(EEG_CHANNEL_NAMES)
    name_str = " ".join(names)

    for band in FeatureExtractor.FREQ_BANDS:
        assert band in name_str, f"band '{band}' missing from feature names"
    print(f"all band names present: {list(FeatureExtractor.FREQ_BANDS.keys())}")


def test_preprocessor_default_params():
    preprocessor = EEGPreprocessor()
    assert preprocessor.l_freq == 0.1
    assert preprocessor.h_freq == 30.0
    assert preprocessor.notch_freq == 50.0
    assert preprocessor.epoch_tmin == -0.2
    assert preprocessor.epoch_tmax == 0.8
    print("EEGPreprocessor default parameters correct")


def test_preprocessor_custom_params():
    preprocessor = EEGPreprocessor(l_freq=1.0, h_freq=40.0)
    assert preprocessor.l_freq == 1.0
    assert preprocessor.h_freq == 40.0
    print("EEGPreprocessor custom parameters accepted")


def test_data_loader_missing_path():
    import tempfile, pathlib
    nonexistent = str(pathlib.Path(tempfile.gettempdir()) / "does_not_exist_lie_detector")
    try:
        EEGDataLoader(nonexistent)
        assert False, "should have raised ValueError"
    except ValueError:
        pass
    print("EEGDataLoader raises ValueError for missing data root")


def test_block_types_keys():
    expected = {'honest_true', 'deceitful_true', 'honest_fake', 'deceitful_fake'}
    assert set(EEGDataLoader.BLOCK_TYPES.values()) == expected, (
        f"unexpected block type values: {set(EEGDataLoader.BLOCK_TYPES.values())}"
    )
    print(f"BLOCK_TYPES values correct: {expected}")


def test_collect_feature_data_empty():
    class FakeLoader:
        participants = ['NONEXISTENT']

    metadata = pd.DataFrame({'ID': ['OTHER'], 'Sex': ['M']})
    real_features = {'p300': {'honest_true': {}, 'deceitful_true': {},
                               'honest_fake': {}, 'deceitful_fake': {}}}
    result = collect_feature_data(FakeLoader(), metadata, real_features, 'p300')
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    print("collect_feature_data returns empty DataFrame when no matching participants")


if __name__ == "__main__":
    tests = [
        test_statistical_feature_shape,
        test_frequency_feature_shape,
        test_combined_feature_shape,
        test_no_nans_in_features,
        test_frequency_features_nonnegative,
        test_feature_names_count_matches_features,
        test_features_vary_across_epochs,
        test_band_names_present_in_feature_names,
        test_preprocessor_default_params,
        test_preprocessor_custom_params,
        test_data_loader_missing_path,
        test_block_types_keys,
        test_normalize_metadata_columns_polish,
        test_normalize_metadata_columns_already_english,
        test_collect_feature_data_empty,
    ]

    print(f"Running lie-detector tests  ({N_CHANNELS} channels, sfreq={SFREQ})\n")

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"✗ {t.__name__}: {e}")
        except Exception as e:
            print(f"✗ {t.__name__}: unexpected error — {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    if passed != len(tests):
        sys.exit(1)
