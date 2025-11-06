from data_loader import EEGDataLoader, get_default_data_path

data_path = get_default_data_path()
loader = EEGDataLoader(str(data_path))

print("\n" + "="*60)
print("PARTICIPANT SUMMARY WITH SEX METADATA")
print("="*60)
summary = loader.create_summary_dataframe()
print(summary[['participant_id', 'sex', 'n_files']].to_string())

print("\n" + "="*60)
print("PARTICIPANTS BY SEX")
print("="*60)
sex_counts = summary['sex'].value_counts()
print(sex_counts)
print(f"\nTotal: {len(summary)} participants")
print(f"Male (M): {sex_counts.get('M', 0)} participants")
print(f"Female (K): {sex_counts.get('K', 0)} participants")
