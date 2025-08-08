import os
import dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any
from huggingface_hub import login
import torch.cuda

from constant.constants import (
    HF_MODELS,
    OPENAI_MODELS,
    FILE_CANDIDATE_POOL_ADDRESS,
    FILE_DESTINATION_ADDRESS,
)
from constant.climate_framework import climate_assessment_prompt, system_prompt

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
# API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
# if API_KEY:
#     login(token=API_KEY)
#     print("Hugging Face authentication successful")

# huggingface_models = [
#     "meta-llama/Meta-Llama-3-8B-Instruct",
#     "Qwen/Qwen2.5-7B-Instruct",
#     "mistralai/Mixtral-8x7B-Instruct-v0.1",
#     "Qwen/Qwen2.5-14B-Instruct",
#     "google/gemma-2-9b-it",
#     "mistralai/Mistral-Small-24B-Instruct-2501"
# ]


def check_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Available: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB used")
    else:
        print("No GPU available")


def initialize_hf_model(model_name: str):
    print(f"Loading Hugging Face model: {model_name}")
    
    # Check GPU memory first
    check_gpu_memory()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Let it auto-detect GPU
        quantization_config=config,
        torch_dtype=torch.float16
    )
    model.gradient_checkpointing_enable()
    
    return model, tokenizer


def generate_hf_answer(prompt: str, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=2000,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if result.startswith(prompt):
        result = result[len(prompt):].strip()
    
    return result


def generate_openai_answer(prompt: str, model_name: str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000
    )
    return response.choices[0].message.content.strip()


def parse_response(response: str):
    lines = response.strip().split('\n')
    parsed_data = {}
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'Region':
                parsed_data['region'] = value
            elif key == 'Exposure':
                parsed_data['exposure_score'] = value
            elif key == 'Sensitivity':
                parsed_data['sensitivity_score'] = value
            elif key == 'Adaptability':
                parsed_data['adaptability_score'] = value
            elif key == 'Temporal_Scale':
                parsed_data['temporal_scale_focus'] = value
            elif key == 'Functional_System':
                parsed_data['functional_system_focus'] = value
            elif key == 'Spatial_Scale':
                parsed_data['spatial_scale_focus'] = value
            elif key == 'Answer':
                parsed_data['answer'] = value
    
    default_keys = [
        'region', 'exposure_score', 'sensitivity_score', 'adaptability_score', 
        'temporal_scale_focus', 'functional_system_focus',
        'spatial_scale_focus', 'answer'
    ]
    
    for key in default_keys:
        if key not in parsed_data:
            parsed_data[key] = 'NA'
    
    parsed_data['full_response'] = response
    return parsed_data


def load_models(selected_models: List[str]):
    loaded_models = {}
    
    for model_id in selected_models:
        if model_id in HF_MODELS:
            model_name = HF_MODELS[model_id]
            model, tokenizer = initialize_hf_model(model_name)
            loaded_models[model_id] = {
                'type': 'hf',
                'name': model_name,
                'model': model,
                'tokenizer': tokenizer
            }
            print(f"Successfully loaded HF model: {model_name}")
        
        elif model_id in OPENAI_MODELS:
            model_name = OPENAI_MODELS[model_id]
            loaded_models[model_id] = {
                'type': 'openai',
                'name': model_name
            }
            print(f"Successfully loaded OpenAI model: {model_name}")
    
    return loaded_models


def generate_answer_with_model(query: str, context: str, model_id: str, loaded_models: Dict):
    prompt = climate_assessment_prompt.format(query=query, context=context)
    
    model_info = loaded_models[model_id]
    
    if model_info['type'] == 'hf':
        response = generate_hf_answer(prompt, model_info['model'], model_info['tokenizer'])
    elif model_info['type'] == 'openai':
        response = generate_openai_answer(prompt, model_info['name'])
    
    parsed_data = parse_response(response)
    return response, parsed_data


def process_single_query(row, model_id: str, loaded_models: Dict, correct_passage_index: int):
    query = row['query']
    passage_column = f'passage_{correct_passage_index}'
    correct_passage = row[passage_column]
    
    response, parsed_data = generate_answer_with_model(query, correct_passage, model_id, loaded_models)
    
    return {
        'query': query,
        'correct_passage_context': correct_passage,
        'model_id': model_id,
        'model_name': loaded_models[model_id]['name'],
        'region': parsed_data['region'],
        'exposure_score': parsed_data['exposure_score'],
        'sensitivity_score': parsed_data['sensitivity_score'],
        'adaptability_score': parsed_data['adaptability_score'],
        'temporal_scale_focus': parsed_data['temporal_scale_focus'],
        'functional_system_focus': parsed_data['functional_system_focus'],
        'spatial_scale_focus': parsed_data['spatial_scale_focus'],
        'answer': parsed_data['answer'],
        'full_response': parsed_data['full_response']
    }


def process_csv_multiple_models(input_file_path: str, output_file_path: str, selected_models: List[str], max_rows=None):
    df = pd.read_csv(input_file_path)
    
    if max_rows is not None:
        df = df.head(max_rows)
    
    loaded_models = load_models(selected_models)
    
    all_results = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
        correct_passage_index = int(row['correct_passage_index'])
        
        for model_id in selected_models:
            if model_id in loaded_models:
                result = process_single_query(row, model_id, loaded_models, correct_passage_index)
                all_results.append(result)
                time.sleep(0.5)
        
        if (index + 1) % 10 == 0:
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(f'{output_file_path}.temp', index=False)    
    results_df = pd.DataFrame(all_results)
    
    text_columns = ['query', 'correct_passage_context', 'answer']
    for col in text_columns:
        if col in results_df.columns:
            results_df[col] = results_df[col].apply(lambda x: f'"{str(x).replace(chr(10), " ").replace(chr(13), " ").strip()}"' if pd.notna(x) else '""')
    
    results_df.to_csv(output_file_path, index=False, escapechar="\\")
    
    temp_file = f'{output_file_path}.temp'
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return results_df


def select_models_interactive():
    for key, value in HF_MODELS.items():
        print(f"{key}: {value}")
    
    print("\nOpenAI models:")
    for key, value in OPENAI_MODELS.items():
        print(f"{key}: {value}")
    
    choice = input("Model selection: ").strip()
    selected = [x.strip() for x in choice.split(',')]
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model ClimateRAG Evaluation")
    parser.add_argument("--models", type=str, help="Comma-separated model IDs (e.g., 1,3,7,8)")
    parser.add_argument("--input_path", type=str, default=FILE_CANDIDATE_POOL_ADDRESS, help="Path to input CSV")
    parser.add_argument("--output_path", type=str, default=FILE_DESTINATION_ADDRESS, help="Path to save output CSV")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional: max rows to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.models:
        selected_models = [x.strip() for x in args.models.split(',')]
    else:
        selected_models = select_models_interactive()
    
    print(f"Selected models: {selected_models}")
    
    results = process_csv_multiple_models(
        input_file_path=args.input_path,
        output_file_path=args.output_path,
        selected_models=selected_models,
        max_rows=args.max_rows
    )
    
