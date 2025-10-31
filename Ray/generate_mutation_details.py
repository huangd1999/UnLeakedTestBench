# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2025-07-16
# Description: Generate mutation details for each mutation in the database.

import os
import re
import json
import sqlite3
from tqdm import tqdm

def get_mutation_code_from_diff(original_code: str, diff: str) -> str:
    original_lines = original_code.splitlines()
    diff_lines = diff.strip().split('\n')
    
    changes = []
    
    i = 0
    while i < len(diff_lines):
        line = diff_lines[i]
        
        # Look for hunk header (e.g., "@@ -43,7 +43,7 @@")
        hunk_match = re.match(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@', line)
        if hunk_match:
            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2))
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4))
            
            i += 1
            
            # Parse the changes in this hunk
            old_lines = []
            new_lines = []
            
            while i < len(diff_lines):
                line = diff_lines[i]
                
                if line.startswith('@@'):
                    # Next hunk, break
                    break
                elif line.startswith('---') or line.startswith('+++'):
                    # File headers, skip
                    i += 1
                    continue
                elif line.startswith('-'):
                    # Removed line
                    old_lines.append(line[1:])
                elif line.startswith('+'):
                    # Added line
                    new_lines.append(line[1:])
                elif line.startswith(' '):
                    # Context line (unchanged)
                    old_lines.append(line[1:])
                    new_lines.append(line[1:])
                
                i += 1
            
            # Store the change information
            changes.append({
                'old_start': old_start,
                'old_count': old_count,
                'new_start': new_start,
                'new_count': new_count,
                'old_lines': old_lines,
                'new_lines': new_lines
            })
        else:
            i += 1
    
    # Apply changes to the original code
    result_lines = original_lines.copy()
    
    # Process changes in reverse order to maintain line numbers
    for change in reversed(changes):
        old_start = change['old_start'] - 1  # Convert to 0-based indexing
        old_count = change['old_count']
        new_lines = change['new_lines']
        
        # Remove the old lines and insert the new lines
        del result_lines[old_start:old_start + old_count]
        result_lines[old_start:old_start] = new_lines
    
    return '\n'.join(result_lines)


def main(base_dir):
    results = []
    for task_dir in tqdm(os.listdir(base_dir)):
        if not task_dir.startswith("task_"):
            continue

        task_id = task_dir.split("_")[1]
        db_path = os.path.join(base_dir, task_dir, "cosmic-ray.sqlite")
        code_path = os.path.join(base_dir, task_dir, "mod.py")

        if not os.path.exists(db_path) or not os.path.exists(code_path):
            continue

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT job_id, operator_name, start_pos_row, start_pos_col, end_pos_row, end_pos_col FROM mutation_specs")
            mutations = cursor.fetchall()

            cursor.execute("SELECT job_id, test_outcome, diff FROM work_results")
            work_results = {row[0]: {"test_outcome": row[1], "diff": row[2]} for row in cursor.fetchall()}

            with open(code_path, "r") as f:
                original_code = f.read()

            mutants_list = []
            for job_id, operator_name, start_row, start_col, end_row, end_col in mutations:
                status = work_results.get(job_id, {}).get("test_outcome", "pending")
                diff = work_results.get(job_id, {}).get("diff", "No diff")

                mutants_list.append({
                    "status": status,
                    "mutation_operator": operator_name,
                    "mutation_diff": diff,
                    "mutation_code": get_mutation_code_from_diff(original_code, diff),
                    "start_line": start_row,
                    "start_column": start_col,
                    "end_line": end_row,
                    "end_column": end_col,
                })

            results.append({
                "task_id": task_id,
                "original_code": original_code,
                "mutants": mutants_list
            })

    with open("new_mutation_details.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    base_dir = "/home/nus_cisco_wp1/Projects/Ray/data/testbench/mutation_5/TestBench_datasetv6"
    main(base_dir)
