import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    script_path = Path(__file__).parent / "implementation" / script_name
    venv_python = Path(__file__).parent / "implementation" / "venv" / "Scripts" / "python.exe"
    
    try:
        result = subprocess.run(
            [str(venv_python), str(script_path)],
            cwd=Path(__file__).parent / "implementation",
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

def run_notebook(notebook_name, description):
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    notebook_path = Path(__file__).parent / "implementation" / notebook_name
    venv_python = Path(__file__).parent / "implementation" / "venv" / "Scripts" / "python.exe"
    
    try:
        result = subprocess.run(
            [str(venv_python), "-m", "jupyter", "nbconvert", 
             "--to", "notebook", 
             "--execute", 
             "--inplace",
             "--ExecutePreprocessor.timeout=600",
             "--ExecutePreprocessor.kernel_name=lie-detector-venv",
             str(notebook_path)],
            cwd=Path(__file__).parent / "implementation",
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"\n{description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError running {notebook_name}: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("EEG Truth and Lie Detection")
    print("="*80)
    
    start_time = time.time()
    
    scripts = [
        ("extract_features.py", "Feature Extraction"),
        ("sex_analysis.py", "Sex-Based Analysis"),
        ("age_analysis.py", "Age-Based Analysis"),
    ]
    
    notebooks = [
        ("notebooks/eda.ipynb", "Exploratory Data Analysis"),
        ("notebooks/baseline_models.ipynb", "Baseline Machine Learning Models"),
        ("notebooks/neural_networks.ipynb", "Neural Network Experiments"),
    ]
    
    results = []
    
    print("\nPhase 1: Running Python Scripts")
    print("-" * 80)
    
    for script_name, description in scripts:
        success = run_script(script_name, description)
        results.append((description, success, "Script"))
        
        if not success:
            print(f"\nWarning: {description} failed.")
        
        time.sleep(1)
    
    print("\n\nPhase 2: Executing Jupyter Notebooks")
    print("-" * 80)
    
    for notebook_name, description in notebooks:
        success = run_notebook(notebook_name, description)
        results.append((description, success, "Notebook"))
        
        if not success:
            print(f"\nWarning: {description} failed.")
        
        time.sleep(1)

if __name__ == "__main__":
    main()
