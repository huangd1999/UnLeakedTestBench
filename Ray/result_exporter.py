# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-07-13

import re
import json
import subprocess
from tqdm.contrib.concurrent import process_map

# import the filtered tasks
def import_filtered_tasks(benchmark_name):
    filtered_tasks = list()
    filtered_tasks_path = f'data/{benchmark_name}_generation/filtered_tasks.json'
    with open(filtered_tasks_path, 'r') as f:
        for task in json.loads(f.read()):
            filtered_tasks.append(f"task_{task['task_id']}")
    return filtered_tasks

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
    
    filtered_tasks = import_filtered_tasks(benchmark_name)
    print(f'[+] ✅ Filtered Tasks: {len(filtered_tasks)}')
    final_tasks = list(set(correct_tasks) & set(filtered_tasks))
    print(f'[+] ✅ Final Tasks: {len(final_tasks)}')
    
    # final_tasks = correct_tasks
    
    surviving_mutants_rate = 0.0

    statistics = process_map(mutation_statistic_wrapper, [benchmark_name]*len(final_tasks), [model_name]*len(final_tasks), [num_test_cases]*len(final_tasks), final_tasks, desc=f"[+] 🔄 Running mutation ({num_test_cases} test cases) statistics...", chunksize=1)
    for statistic in statistics:
        print(f"[+] {statistic}")
        surviving_mutants_rate += statistic["surviving_mutants_rate"]
    
    surviving_mutants_rate = (surviving_mutants_rate / len(correct_tasks)) if len(correct_tasks) > 0 else 0.0
    print(f'[+] ✅ Surviving Mutants Rate: {surviving_mutants_rate:.2%} \n')

    return surviving_mutants_rate


if __name__ == '__main__':
    benchmark_name = 'testbench'
    num_test_cases = 5
    baseline_test_cases = 5
    model_generation_file_path = 'data/testbench_generation/TestBench_CodeLlama-7b-Instruct-hf_mutants.jsonl'
    surviving_mutants_rate = mutation_statistic(benchmark_name, model_generation_file_path, num_test_cases, baseline_test_cases)
