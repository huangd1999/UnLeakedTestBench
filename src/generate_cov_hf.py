import os
import re
import json
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import torch

# 导入vLLM相关库
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from data_utils import read_jsonl, write_jsonl, add_lineno

def extract_function_names_from_completion(completion: str) -> list:
    """Extract function names from the completion code."""
    # Regular expression to match function definitions (ignoring indented functions)
    function_pattern = r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    
    # Find all matches
    function_names = re.findall(function_pattern, completion, re.MULTILINE)
    return function_names

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='leetcode')
    parser.add_argument("--model", type=str, default='codellama/CodeLlama-7b-Instruct-hf')
    parser.add_argument("--num_tests", type=int, default=20, help='number of tests generated per program')
    parser.add_argument("--temperature", type=float, default=0)  # 最小设置为0.01，避免数值错误
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=128, help='batch size for inference')
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help='number of GPUs for tensor parallelism')
    parser.add_argument("--max_context_length", type=int, default=4096, help='maximum context length for truncation')
    return parser.parse_args()

model_list = [
    # "codellama/CodeLlama-7b-Instruct-hf",
    # "ByteDance-Seed/Seed-Coder-8B-Instruct",
    "google/gemma-3-4b-it", 
    # "google/gemma-3-12b-it", 
    # "google/gemma-3-27b-it",
    "Qwen/Qwen2.5-Coder-7B-Instruct", 
    # "Qwen/Qwen2.5-Coder-14B-Instruct", 
    # "Qwen/Qwen2.5-Coder-32B-Instruct",
    # 'deepseek-ai/deepseek-coder-1.3b-instruct', 
    'deepseek-ai/deepseek-coder-6.7b-instruct', 
    'deepseek-ai/deepseek-coder-33b-instruct',
    # "microsoft/Phi-4-mini-instruct", 
    # "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    # "google/gemma-2-9b-it",
    # "google/gemma-2-27b-it",
    # "meta-llama/CodeLlama-70b-Instruct-hf",
]

def truncate_conversation(messages, tokenizer, max_length):
    """截断对话历史，确保总长度不超过 max_length"""
    # 尝试获取系统提示和最后一条用户消息
    system_message = next((m for m in messages if m["role"] == "system"), None)
    last_user_message = next((m for m in reversed(messages) if m["role"] == "user"), None)
    
    # 保留系统消息和最后的用户消息，舍弃中间的历史
    truncated_messages = []
    if system_message:
        truncated_messages.append(system_message)
    if last_user_message:
        truncated_messages.append(last_user_message)
    
    # 尝试添加一条最近的助手消息（如果存在），以保持上下文连贯性
    recent_assistant_messages = [m for m in reversed(messages) if m["role"] == "assistant"]
    if recent_assistant_messages and len(recent_assistant_messages) > 0:
        truncated_messages.insert(1, recent_assistant_messages[0])
    
    return truncated_messages

def format_chat_template(tokenizer, messages):
    """Format chat messages to model-specific prompt format"""
    try:
        # 尝试使用 apply_chat_template (Transformers >= 4.35.0)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        # 回退方案：简单格式化消息
        formatted_prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
        formatted_prompt += "Assistant: "
        return formatted_prompt

def prepare_prompts_for_batch(data_batch, prompt_template, system_message, tokenizer):
    """Prepare multiple prompts for batch inference"""
    prompts = []
    
    for data in data_batch:
        try:
            func_names = extract_function_names_from_completion(data["code"])
            if func_names:
                func_name = func_names[0]
                desc = data['prompt']
                code = data['code']
                task_id = data['task_id']

                prompt = prompt_template.format(lang='python', program=code, description=desc, func_name=func_name)
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
                formatted_prompt = format_chat_template(tokenizer, messages)
                prompts.append((formatted_prompt, func_name, code, desc, task_id))
            else: 
                # Skip if no function name found
                print(f"No function name found in the code")
                continue
        except Exception as e:
            print(f"Error preparing prompt: {e}")
            continue
            
    return prompts


def testgeneration_vllm_batch(prepared_prompts, llm, sampling_params, tokenizer, max_tokens=4096):
    """Generate test cases in batch using vLLM with prompt truncation"""
    if not prepared_prompts:
        return []
    
    # 截断过长提示词
    truncated_prompts = []
    for prompt_data in prepared_prompts:
        prompt_text, func_name, code, desc, task_id = prompt_data
        
        # 使用tokenizer计算token数量
        tokens = tokenizer.encode(prompt_text)
        
        # 如果超过最大长度，截取最后max_tokens个token
        if len(tokens) > max_tokens:
            print(f"Truncating prompt from {len(tokens)} to {max_tokens} tokens")
            truncated_tokens = tokens[-max_tokens:]
            truncated_text = tokenizer.decode(truncated_tokens)
            truncated_prompts.append((truncated_text, func_name, code, desc, task_id))
        else:
            truncated_prompts.append(prompt_data)
    
    # Extract just the prompt texts
    prompt_texts = [p[0] for p in truncated_prompts]
    
    # Run batch inference with vLLM
    outputs = llm.generate(prompt_texts, sampling_params)
    
    # Create result dictionary with generated text and metadata
    results = []
    for i, output in enumerate(outputs):
        _, func_name, code, desc, task_id = truncated_prompts[i]
        generated_test = output.outputs[0].text
        if "</think>" in generated_test:
            generated_test = generated_test.split("</think>")[1]
        results.append({
            'func_name': func_name,
            'code': code,
            'test': generated_test,
            'prompt': desc,
            'task_id':task_id
        })
    
    return results

def testgeneration_multiround_vllm(args, dataset, prompt_template, system_message, tokenizer, llm):
    """Generate multiple tests for each sample using batched processing with vLLM"""
    all_results = []
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=1.0,
    )
    
    # First round: generate initial test for each sample
    for batch_start in tqdm(range(0, len(dataset), args.batch_size), desc="Processing initial batch"):
        batch_end = min(batch_start + args.batch_size, len(dataset))
        data_batch = dataset[batch_start:batch_end]
        
        prepared_prompts = prepare_prompts_for_batch(data_batch, prompt_template, system_message, tokenizer)
        batch_results = testgeneration_vllm_batch(prepared_prompts, llm, sampling_params,tokenizer)
        
        # Initialize the results structure with the first test for each sample
        for result in batch_results:
            all_results.append({
                'func_name': result['func_name'],
                'code': result['code'],
                'tests': [result['test']],
                'prompt': result['prompt'],
                'task_id':result['task_id']
            })
    
    # For additional test rounds
    template_append = "Generate another test method for the function under test. Your answer must be different from previously-generated test cases, and should cover different statements and branches."
    
    # Generate remaining tests (num_tests - 1)
    for test_round in range(1, args.num_tests):
        print(f"Starting test generation round {test_round+1}/{args.num_tests}")
        
        # Process each result entry to generate a new test
        for batch_start in tqdm(range(0, len(all_results), args.batch_size), desc=f"Round {test_round+1}"):
            batch_end = min(batch_start + args.batch_size, len(all_results))
            current_batch = all_results[batch_start:batch_end]
            
            prepared_prompts = []
            for result in current_batch:
                # Create conversation history with all previous tests
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_template.format(
                        lang='python', 
                        program=result['code'], 
                        description=result['prompt'], 
                        func_name=result['func_name'],
                    )}
                ]
                
                # Add conversation history with previous tests
                for prev_test in result['tests']:
                    messages.append({"role": "assistant", "content": prev_test})
                    messages.append({"role": "user", "content": template_append})
                
                # 检查并截断对话历史，如果太长
                if len(tokenizer.encode(" ".join([m["content"] for m in messages]))) > args.max_context_length:
                    messages = truncate_conversation(messages, tokenizer, args.max_context_length)
                
                # Format as prompt
                formatted_prompt = format_chat_template(tokenizer, messages)
                prepared_prompts.append((
                    formatted_prompt, 
                    result['func_name'], 
                    result['code'], 
                    result['prompt'],
                    result['task_id']
                ))
                
            # Run batch inference for this round
            if prepared_prompts:
                round_results = testgeneration_vllm_batch(prepared_prompts, llm, sampling_params,tokenizer)
                
                # Update results with new tests
                for i, new_result in enumerate(round_results):
                    idx = batch_start + i
                    if idx < len(all_results):  # Safety check
                        all_results[idx]['tests'].append(new_result['test'])
                
    return all_results

if __name__=='__main__':
    args = parse_args()
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    access_token = os.getenv("HUGGINGFACE_TOKEN")
    with open("../dataset/ULT.jsonl","r") as f:
        dataset = json.load(f)

    prompt_template = open('prompt/template_base.txt').read()
    system_template = open('prompt/system.txt').read()
    system_message = system_template.format(lang='python')

    for model_name in model_list:
        args.model = model_name
        model_abbrv = args.model.split('/')[-1]
        print('='*50)
        print(f'Model: {model_abbrv}')
        print('='*50)
        if os.path.exists(output_dir / f'TestBench_{model_abbrv}_{args.num_tests}_full.jsonl'):
            print(f"Results for {model_abbrv} already exist, skipping...")
            continue
        try:
            # 加载 tokenizer 用于格式化提示词
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, 
                use_auth_token=access_token,
                trust_remote_code=True
            )
            
            # # 获取模型上下文窗口长度
            model_context_length = 16384  # 默认设置较大值
            if hasattr(tokenizer, 'model_max_length'):
                model_context_length = min(tokenizer.model_max_length,model_context_length)  # 避免极端值
            # if args.model =="meta-llama/CodeLlama-70b-Instruct-hf":
            #     model_context_length = 4096
            # elif args.model.startswith("google/gemma"):
            # model_context_length = 4096        

            # 初始化 vLLM 实例
            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,  # 设置张量并行大小
                trust_remote_code=True,
                dtype="float16",  # 使用 float16 而不是 bfloat16，以优化速度
                # gpu_memory_utilization=0.95,  # 默认 0.9
                max_model_len=model_context_length,  # 使用最大可能的上下文长度
                quantization="awq" if Path(f"./quantized/{model_abbrv}_awq").exists() else None  # 可选择量化模型
            )

            data_size = len(dataset)
            print('Number of samples:', data_size)
            
            # 使用 vLLM 进行批量测试生成
            testing_results = testgeneration_multiround_vllm(
                args, dataset, prompt_template, system_message, tokenizer, llm
            )
            
            # 保存结果
            write_jsonl(testing_results, output_dir / f'TestBench_{model_abbrv}_{args.num_tests}_full.jsonl')
            print(f"Results saved to {output_dir}/TestBench_{model_abbrv}_{args.num_tests}_full.jsonl")
            # write_jsonl(testing_results, output_dir / f'TestBench_{model_abbrv}.jsonl')
            # print(f"Results saved to {output_dir}/TestBench_{model_abbrv}.jsonl")

        except Exception as e:
            print(f"Error during test generation with model {model_abbrv}: {e}")
        
        # 清理以释放 GPU 内存
        if 'llm' in locals():
            del llm
        if 'tokenizer' in locals():
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"Completed processing for model: {model_abbrv}")
