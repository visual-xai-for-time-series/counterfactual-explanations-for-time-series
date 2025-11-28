#!/usr/bin/env python3
"""
Run All Examples and Metrics Evaluation Script

This script runs all the counterfactual explanation examples and the comprehensive
metrics evaluation in sequence. It provides a convenient way to execute the entire
pipeline and generate all results.

Features:
- Runs all three main examples (Arabic Digits, Vibration, FordA)
- Executes comprehensive metrics evaluation
- Provides timing information for each step
- Handles errors gracefully and continues execution
- Generates summary report of all results
"""

import os
import sys
import time
import subprocess
import traceback
from datetime import datetime

# Add parent directory to path
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../')

def print_header(title):
    """Print a formatted header for each section."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "-"*60)
    print(f" {title}")
    print("-"*60)

def run_script(script_name, description):
    """
    Run a Python script and return success status, duration, and any error.
    
    Args:
        script_name (str): Name of the script to run
        description (str): Human-readable description of the script
    
    Returns:
        tuple: (success, duration, error_message)
    """
    print_section(f"Running {description}")
    print(f"Script: {script_name}")
    
    start_time = time.time()
    
    try:
        # Get the Python executable path for the virtual environment
        python_executable = sys.executable
        
        # Run the script
        result = subprocess.run(
            [python_executable, script_name],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ SUCCESS - Completed in {duration:.2f} seconds")
            if result.stdout:
                print("Output summary:")
                # Print last few lines of output for summary
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-5:]:
                    print(f"  {line}")
            return True, duration, None
        else:
            print(f"✗ FAILED - Error after {duration:.2f} seconds")
            error_msg = result.stderr if result.stderr else "Unknown error"
            print(f"Error: {error_msg}")
            return False, duration, error_msg
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        error_msg = f"Script timed out after {duration:.2f} seconds"
        print(f"✗ TIMEOUT - {error_msg}")
        return False, duration, error_msg
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Unexpected error: {str(e)}"
        print(f"✗ ERROR - {error_msg}")
        return False, duration, error_msg

def check_requirements():
    """Check if required files and dependencies exist."""
    print_section("Checking Requirements")
    
    # Check if example scripts exist
    required_scripts = [
        'example_multivariate.py',
        'example_univariate.py',
        'metrics_evaluation_example.py'
    ]
    
    missing_scripts = []
    for script in required_scripts:
        script_full_path = os.path.join(script_path, script)
        if not os.path.exists(script_full_path):
            missing_scripts.append(script)
            print(f"✗ Missing: {script}")
        else:
            print(f"✓ Found: {script}")
    
    if missing_scripts:
        print(f"\nError: Missing required scripts: {missing_scripts}")
        return False
    
    # Check if models directory exists
    models_dir = os.path.join(script_path, '..', 'models')
    if not os.path.exists(models_dir):
        print(f"✓ Models directory will be created: {models_dir}")
    else:
        print(f"✓ Models directory exists: {models_dir}")
    
    return True

def main():
    """Main execution function."""
    start_time = time.time()
    
    print_header("Counterfactual Explanations - Complete Pipeline Execution")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check requirements
    if not check_requirements():
        print("\n✗ Requirements check failed. Exiting.")
        return 1
    
    # Results tracking
    results = []
    
    # List of scripts to run
    scripts_to_run = [
        ('example_multivariate.py', 'Arabic Digits Counterfactual Examples'),
        ('example_univariate.py', 'FordA Dataset Counterfactual Examples'),
        ('metrics_evaluation_example.py', 'Comprehensive Metrics Evaluation')
    ]
    
    print_section("Execution Plan")
    print("The following scripts will be executed in order:")
    for i, (script, description) in enumerate(scripts_to_run, 1):
        print(f"  {i}. {script} - {description}")
    
    # Execute each script
    print_header("Script Execution")
    
    for script, description in scripts_to_run:
        script_full_path = os.path.join(script_path, script)
        success, duration, error = run_script(script_full_path, description)
        
        results.append({
            'script': script,
            'description': description,
            'success': success,
            'duration': duration,
            'error': error
        })
        
        # Short pause between scripts
        if script != scripts_to_run[-1][0]:  # Not the last script
            print("Waiting 2 seconds before next script...")
            time.sleep(2)
    
    # Generate summary report
    total_duration = time.time() - start_time
    print_header("Execution Summary")
    
    print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Results table
    print("\nResults Summary:")
    print(f"{'Script':<35} {'Status':<10} {'Duration':<12} {'Description'}")
    print("-" * 80)
    
    successful_count = 0
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        duration_str = f"{result['duration']:.1f}s"
        
        print(f"{result['script']:<35} {status:<10} {duration_str:<12} {result['description']}")
        
        if result['success']:
            successful_count += 1
        elif result['error']:
            print(f"  Error: {result['error'][:60]}...")
    
    print("-" * 80)
    print(f"Success rate: {successful_count}/{len(results)} ({100*successful_count/len(results):.1f}%)")
    
    # List generated files
    print_section("Generated Files")
    generated_files = [
        'counterfactuals_arabic_digits.png',
        'counterfactuals_forda.png'
    ]
    
    for filename in generated_files:
        filepath = os.path.join(os.path.dirname(script_path), filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"✓ {filename} ({file_size/1024:.1f} KB)")
        else:
            print(f"✗ {filename} (not found)")
    
    # Check for model files
    models_dir = os.path.join(script_path, '..', 'models')
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if model_files:
            print(f"\nTrained models in {models_dir}:")
            for model_file in model_files:
                filepath = os.path.join(models_dir, model_file)
                file_size = os.path.getsize(filepath)
                print(f"✓ {model_file} ({file_size/1024/1024:.1f} MB)")
    
    # Final status
    if successful_count == len(results):
        print_header("🎉 ALL SCRIPTS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print_header(f"⚠️  {len(results) - successful_count} SCRIPTS FAILED")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
