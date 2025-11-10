import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    script_path = Path(__file__).parent / script_name
    venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
    
    try:
        result = subprocess.run(
            [str(venv_python), str(script_path)],
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True,
            check=True
        )
        
        print(f"\n{description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: {e}")
        return False
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("EEG Truth and Lie Detection")
    print("="*80)
    
    start_time = time.time()
    
    steps = [
        ("extract_features.py", "Feature Extraction"),
        ("sex_analysis.py", "Sex-Based Analysis"),
        ("age_analysis.py", "Age-Based Analysis)"),
    ]
    
    results = []
    
    for script_name, description in steps:
        success = run_script(script_name, description)
        results.append((description, success))
        
        if not success:
            print(f"\nWarning: {description} failed. Continuing with next step...")
        
        time.sleep(1)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print("\nExecution Summary:")
    for description, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {status}: {description}")

if __name__ == "__main__":
    main()
