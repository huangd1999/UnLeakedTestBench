import ast
import json
import radon
from tqdm import tqdm
from datasets import load_dataset
from radon.visitors import ComplexityVisitor


dataset_name = "kjain14/testgeneval"
split = "test"
ds = load_dataset(dataset_name, split=split)


def calculate_cyclomatic_complexity(code_src):
    try:
        ast.parse(code_src)
        complexity_list = list()
        visitor = ComplexityVisitor.from_code(code_src)
        
        # Split the source code into lines
        code_lines = code_src.splitlines()
        
        complexity_list = [{
            'func': func,
            'complexity': func.complexity,
            'name': func.name,
            'start_line': func.lineno,
            'end_line': func.endline,
            'code': '\n'.join(code_lines[func.lineno - 1:func.endline])
        } for func in visitor.functions]
        
        return complexity_list
    except Exception:
        return []


results = list()

for instance in tqdm(ds):
    instance_id = instance['id']
    code_src = instance['code_src']
    cyclomatic_complexity = calculate_cyclomatic_complexity(code_src)
    results.append({
        'id': instance_id,
        'cyclomatic_complexity_list': cyclomatic_complexity
    })


with open('cyclomatic_complexity.jsonl', 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')





