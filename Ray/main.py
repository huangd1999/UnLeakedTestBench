# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-05-27

import re
import os
import json
import string
import random
import shutil
import tempfile
import datetime
import argparse
import subprocess
from tqdm import tqdm
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

toml_template = """
[cosmic-ray]
module-path = "mod.py"
timeout = {timeout}
excluded-modules = []
test-command = "pytest test.py"

[cosmic-ray.distributor]
name = "local"
"""

code_import = """
import os
import re
import math
import numpy
import pandas
import pytest
import random
import string
import warnings
import datetime
import traceback
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Set, FrozenSet, Sequence, Iterable, Generator, Callable
"""

def rename_test_functions(test_code):
    test_counter = 0
    new_lines = list()
    pattern = re.compile(r"(\s*def\s+)(test_)(.*)")
    test_code_lines = test_code.split('\n')
    
    for line in test_code_lines:
        match = pattern.match(line)
        if match:
            test_counter += 1
            new_lines.append(f"{match.group(1)}test_{test_counter}_{match.group(3)}")
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)

def parse_pytest_output(output: str) -> dict:
    """
    Parses the stdout of a `pytest --cov` run to extract key metrics.

    Args:
        output: The string output from the pytest command.

    Returns:
        A dictionary containing test results and coverage.
    """
    # 🎯 Default values
    total_coverage = 0
    passed_count = 0
    failed_count = 0

    # Regex to find the total coverage percentage from the 'TOTAL' line
    coverage_match = re.search(r"^TOTAL\s+.*\s+(\d+)%$", output, re.MULTILINE)
    if coverage_match:
        total_coverage = int(coverage_match.group(1))

    # Regex to find the numbers in the final summary line
    # e.g., "===... 4 failed, 1 passed in 9.85s ...==="
    summary_line_match = re.search(
        r"=+ (.*) in .*s =+", output
    )
    if summary_line_match:
        summary_text = summary_line_match.group(1)
        
        passed_match = re.search(r"(\d+)\s+passed", summary_text)
        if passed_match:
            passed_count = int(passed_match.group(1))
            
        failed_match = re.search(r"(\d+)\s+failed", summary_text)
        if failed_match:
            failed_count = int(failed_match.group(1))

    # --- Calculations ---
    total_tests = passed_count + failed_count
    pass_rate = 0.0
    if total_tests > 0:
        pass_rate = (passed_count / total_tests) * 100

    return {
        "total_coverage_percent": total_coverage,
        "pass_rate_percent": round(pass_rate, 2),
        "passed_tests": passed_count,
        "failed_tests": failed_count,
        "total_tests": total_tests,
    }

# Initialization 
def cosmic_ray_init(benchmark_name, model_generation_file, num_test_cases=5, timeout=1, num_samples=100):
    model_name = model_generation_file.split('/')[-1].split('.')[0]

    if os.path.exists(f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}'):
        print(f"[+] 🧹 Cleaning up existing files in {model_name}...")
        try:
            shutil.rmtree(f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}')
        except PermissionError:
            print(f"[-] PermissionError: {f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}'}")
        
    print(f"[+] 📂 Creating new directory {model_name}...")
    os.makedirs(f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}')

    with open(model_generation_file, 'r') as data_handler:
        # For [json] file
        raw_data = json.loads(data_handler.read())[:num_samples]

        # For [jsonl] file
        # raw_data = [json.loads(line) for line in data_handler.readlines()[:num_samples]]
    print(f"[+] ✅ Raw data: {len(raw_data)}")

    for idx, instance in tqdm(enumerate(raw_data), desc="[+] 💾 Processing raw data"):
        os.makedirs(f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}/task_{idx}')

        # create 'mod.py'
        with open(f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}/task_{idx}/mod.py', 'w') as f:
            mod_code = ''
            mod_code += code_import + '\n\n'
            mod_code += instance['code'] + '\n\n'
            mod_code = rename_test_functions(mod_code)
            f.write(mod_code)

        # create 'test.py'
        with open(f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}/task_{idx}/test.py', 'w') as f:
            test_code = code_import + '\n\n' + 'from mod import *' + '\n\n'
            for test in instance['tests'][:num_test_cases]:
                test_code += f'{test}\n\n'
            # test_code += "\n\n" + "#" * 100 + "\n\n"
            f.write(test_code)         

        # create 'toml'
        with open(f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}/task_{idx}/cosmic-ray.toml', 'w') as f:
            f.write(toml_template.format(model_name=model_name, task_id=idx, timeout=timeout))

def cosmic_ray_setup_wrapper(benchmark_name, model_name, task_id, num_test_cases=5):
    working_dir = f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}/{task_id}'
    
    # Initialize Cosmic-Ray Config
    try:
        subprocess.run(['cosmic-ray', 'init', 'cosmic-ray.toml', 'cosmic-ray.sqlite'], cwd=working_dir, check=True)
    except Exception as e:
        print(f'[-] Initialize Cosmic-Ray Error: {e}')
        return False

    # Run Cosmic-Ray Baseline
    try:
        subprocess.run(['cosmic-ray', 'baseline', 'cosmic-ray.toml'], cwd=working_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60*num_test_cases)
        return True
    except Exception as e:
        return False

def cosmic_ray_setup(benchmark_name, model_generation_file, num_test_cases=5):
    model_name = model_generation_file.split('/')[-1].split('.')[0]
    total_tasks = list()
    correct_tasks = list()
    
    if os.path.exists(f'data/{benchmark_name}/correct_tasks_tc_{num_test_cases}_{model_name}'):
        print(f'[+] ✅ data/{benchmark_name}/correct_tasks_tc_{num_test_cases}_{model_name} (Already exists)')
        return

    for file_name in tqdm(os.listdir(f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}'), desc="[+] ⏳ Filtering baseline tasks"):
        if file_name.startswith('task_'):
            total_tasks.append(file_name)
        
    task_results = process_map(cosmic_ray_setup_wrapper, [benchmark_name]*len(total_tasks), [model_name]*len(total_tasks), total_tasks, [num_test_cases]*len(total_tasks), desc="[+] 🔄 Initialize Cosmic-Ray Mutation", chunksize=1)
    
    # Save correct tasks
    with open(f'data/{benchmark_name}/correct_tasks_tc_{num_test_cases}_{model_name}', 'w') as f:
        for task_id, result in zip(total_tasks, task_results):
            if result:
                correct_tasks.append(task_id)
                f.write(f'{task_id}\n')

    print(f'[+] ✅ Correct Tasks: {len(total_tasks)} -> {len(correct_tasks)} (Convert Rate: {len(correct_tasks) / len(total_tasks):.2%})')

def cosmic_ray_status(benchmark_name, model_name, task, num_test_cases):
    try:
        cosmic_ray_path = f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}/{task}/cosmic-ray.sqlite'
        response = subprocess.run(['cr-report', cosmic_ray_path, '--show-pending'], check=True, capture_output=True, text=True)
    except Exception as e:
        print(f'[-] Error @ [{cosmic_ray_path}]: {e}')
        return (False, 0, 0)
    
    total_jobs_match = re.search(r"total jobs:\s*(\d+)", response.stdout)
    completed_jobs_match = re.search(r"complete:\s*(\d+)\s*\(", response.stdout)

    if total_jobs_match and completed_jobs_match:
        total_jobs_number = int(total_jobs_match.group(1))
        completed_jobs_number = int(completed_jobs_match.group(1))
        # print(f"[+] Task {task}: Total jobs: {total_jobs_number}, Completed jobs: {completed_jobs_number}")
        if total_jobs_number == 0: return (True, 0, 0)
        return (completed_jobs_number == total_jobs_number, total_jobs_number, completed_jobs_number)
    else:
        return (False, 0, 0)
    
def mutation_status(benchmark_name, model_generation_file, num_test_cases):
    model_name = model_generation_file.split('/')[-1].split('.')[0]
    correct_tasks = list()
    correct_tasks_path = f'data/{benchmark_name}/correct_tasks_tc_{num_test_cases}_{model_name}'
    
    with open(correct_tasks_path, 'r') as f:
        for line in f.readlines():
            correct_tasks.append(line.strip())
    print(f'[+] ✅ Correct Tasks: {len(correct_tasks)}')
    
    for task in correct_tasks:
        completed, total_jobs_number, completed_jobs_number = cosmic_ray_status(benchmark_name, model_name, task, num_test_cases)
        if completed: 
            print(f'[+] Task {task}: Completed ({completed_jobs_number}/{total_jobs_number})')
        else: 
            print(f'[-] Task {task}: Incompleted ({completed_jobs_number}/{total_jobs_number})')

def mutation_run_wrapper(benchmark_name, model_name, num_test_cases, task):
    # cosmic-ray exec tutorial.toml tutorial.sqlite
    completed, _, _ = cosmic_ray_status(benchmark_name, model_name, task, num_test_cases)
    if completed: return

    # print(f"[+] Task {task}: Running mutations")
    working_dir = f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}/{task}'
    try:
        subprocess.run(['cosmic-ray', 'exec', f'cosmic-ray.toml', f'cosmic-ray.sqlite'], cwd=working_dir, check=True, timeout=360*num_test_cases)
    except subprocess.TimeoutExpired as e:
        # print(f'[-] mutation_run_wrapper, Timeout: {e}')
        pass
    except Exception as e:
        print(f'[-] mutation_run_wrapper, Error: {e}')

def mutation_run(benchmark_name, model_generation_file, num_test_cases):
    model_name = model_generation_file.split('/')[-1].split('.')[0]
    correct_tasks = list()
    correct_tasks_path = f'data/{benchmark_name}/correct_tasks_tc_5_{model_name}'
    
    with open(correct_tasks_path, 'r') as f:
        for line in f.readlines():
            correct_tasks.append(line.strip())
    print(f'[+] ✅ Correct Tasks: {len(correct_tasks)}')

    print("================================================")
    print(f'[+] ⏱️ Start time: {datetime.datetime.now()}')
    process_map(mutation_run_wrapper, [benchmark_name]*len(correct_tasks), [model_name]*len(correct_tasks), [num_test_cases]*len(correct_tasks), correct_tasks, desc="[+] 🔮 Running mutations...")
    print(f'[+] ⏱️ End time: {datetime.datetime.now()}')

def mutation_statistic_wrapper(benchmark_name, model_name, num_test_cases, task):
    working_dir = f'data/{benchmark_name}/mutation_{num_test_cases}/{model_name}/{task}'

    statistic_info = {
        "task": task,
        "complete_rate": 0.0,
        "surviving_mutants_rate": 0.0,
        "total_jobs_number": 0,
        "completed_jobs_number": 0,
        "surviving_mutants_number": 0
    }

    try:
        response = subprocess.run(['cr-report', f'cosmic-ray.sqlite', '--show-pending'], cwd=working_dir, check=True, capture_output=True, text=True)
    except Exception as e:
        print(f'[-] Error @ [{working_dir}]: {e}')
        return statistic_info

    total_jobs_match = re.search(r"total jobs:\s*(\d+)", response.stdout)
    completed_jobs_match = re.search(r"complete:\s*(\d+)\s*\(", response.stdout)
    surviving_mutants_match = re.search(r"surviving mutants:\s*(\d+)\s*\(", response.stdout)

    if total_jobs_match:
        total_jobs_number = int(total_jobs_match.group(1))
        statistic_info["total_jobs_number"] = total_jobs_number

    if completed_jobs_match:
        completed_jobs_number = int(completed_jobs_match.group(1))
        statistic_info["completed_jobs_number"] = completed_jobs_number
        
    if surviving_mutants_match:
        surviving_mutants_number = int(surviving_mutants_match.group(1))
        statistic_info["surviving_mutants_number"] = surviving_mutants_number
    
    statistic_info['complete_rate'] = statistic_info['completed_jobs_number'] / statistic_info["total_jobs_number"] if statistic_info["total_jobs_number"] > 0 else 0
    statistic_info['surviving_mutants_rate'] = (statistic_info['surviving_mutants_number'] / statistic_info['completed_jobs_number']) if statistic_info['completed_jobs_number'] > 0 else 0

    return statistic_info

def mutation_statistic(benchmark_name, model_generation_file, num_test_cases, baseline_test_cases=5):
    model_name = model_generation_file.split('/')[-1].split('.')[0]
    correct_tasks = list()
    correct_tasks_path = f'data/{benchmark_name}/correct_tasks_tc_{baseline_test_cases}_{model_name}'
    
    with open(correct_tasks_path, 'r') as f:
        for line in f.readlines():  
            correct_tasks.append(line.strip())
    print(f'[+] ✅ Correct Tasks: {len(correct_tasks)}')
    
    surviving_mutants_rate = 0.0

    statistics = process_map(mutation_statistic_wrapper, [benchmark_name]*len(correct_tasks), [model_name]*len(correct_tasks), [num_test_cases]*len(correct_tasks), correct_tasks, desc=f"[+] 🔄 Running mutation ({num_test_cases} test cases) statistics...", chunksize=1)
    for statistic in statistics:
        print(f"[+] {statistic}")
        surviving_mutants_rate += statistic["surviving_mutants_rate"]
    
    surviving_mutants_rate = (surviving_mutants_rate / len(correct_tasks)) if len(correct_tasks) > 0 else 0.0
    print(f'[+] ✅ Surviving Mutants Rate: {surviving_mutants_rate:.2%} \n')

    return surviving_mutants_rate

def pytest_run_wrapper(benchmark_name, model_name, task_id):
    test_file_path = f'data/{benchmark_name}_mods/{model_name}/{task_id}/test.py'
    source_code_path = f'data/{benchmark_name}_mods/{model_name}/{task_id}'
    try:
        # temporary dictionary to execute pytest
        with tempfile.TemporaryDirectory() as temp_dir:
            abs_test_file_path = os.path.abspath(test_file_path)
            abs_source_code_path = os.path.abspath(source_code_path)
            result = subprocess.run(['pytest', abs_test_file_path, f'--cov={abs_source_code_path}', '--cov-branch'], cwd=temp_dir, capture_output=True, text=True, timeout=30)
            result_dict = parse_pytest_output(result.stdout)
        return {'model_name': model_name, 'task': task_id, 'result': result_dict, "status": "success"}
    except Exception as e:
        return {'model_name': model_name, 'task': task_id, 'result': None, "status": "error"}

def pytest_run(benchmark_name, model_name):
    tasks = list()
    
    for task in os.listdir(f'data/{benchmark_name}_mods/{model_name}'):
        if task.startswith('task_'):
            tasks.append((model_name, task))
            
    results = process_map(pytest_run_wrapper, tasks, desc="[+] 🔄 Running pytest", chunksize=1)
    
    with open(f'data/{benchmark_name}_mods/{model_name}/results.jsonl', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", type=str, default='testbench')
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--mode", type=str, default='all')
    args = parser.parse_args()

    # statistic_info = defaultdict(dict)
    # for num_test_cases in [5,2,1]:
    #     print(f"[+] =============================== Processing {args.benchmark_name} {num_test_cases} test cases ================================")
    #     for model_generation_file_path in tqdm(os.listdir(f"data/{args.benchmark_name}_generation"), desc="[+] 🔄 Processing models"):
    #         model_generation_file_path = f"data/{args.benchmark_name}_generation/{model_generation_file_path}"
    #         print(f"[+] Processing {model_generation_file_path}")
            
    #         if args.mode in ['cosmic_ray_init', 'all']:
    #             cosmic_ray_init(args.benchmark_name, model_generation_file_path, timeout=10, num_samples=args.num_samples, num_test_cases=num_test_cases)
    #         if args.mode in ['cosmic_ray_setup', 'all']:
    #             cosmic_ray_setup(args.benchmark_name, model_generation_file_path, num_test_cases=num_test_cases)            
    #         if args.mode in ['mutation_status', 'all']:
    #             mutation_status(args.benchmark_name, model_generation_file_path, num_test_cases=num_test_cases)
    #         if args.mode in ['mutation_run', 'all']:
    #             mutation_run(args.benchmark_name, model_generation_file_path, num_test_cases=num_test_cases)
    #         if args.mode in ['mutation_statistic', 'all']:
    #             surviving_mutants_rate = mutation_statistic(args.benchmark_name, model_generation_file_path, num_test_cases=num_test_cases)
    #             statistic_info[model_generation_file_path][num_test_cases] = surviving_mutants_rate
    # print(statistic_info)
    
    for num_test_cases in [5,2,1]:
        # model_generation_file_path = 'data/testeval_generation/totalcov_Seed-Coder-8B-Instruct_results.jsonl'
        # model_generation_file_path = 'data/testbench_generation/TestBench_datasetv4.jsonl'
        # model_generation_file_path = 'data/testbench_generation/TestBench_gpt-4o_1_0.2_format.jsonl'
        # model_generation_file_path = 'data/testbench_generation/TestBench_CodeLlama-7b-Instruct-hf_mutants.jsonl'
        model_generation_file_path = 'data/testbench_generation/TestBench_datasetv6.jsonl'

        # cosmic_ray_init(args.benchmark_name, model_generation_file_path, timeout=10, num_samples=args.num_samples, num_test_cases=num_test_cases)
        # cosmic_ray_setup(args.benchmark_name, model_generation_file_path, num_test_cases=num_test_cases)
        # mutation_status(args.benchmark_name, model_generation_file_path, num_test_cases=num_test_cases)
        # mutation_run(args.benchmark_name, model_generation_file_path, num_test_cases)
        mutation_statistic(args.benchmark_name, model_generation_file_path, num_test_cases, baseline_test_cases=5)

