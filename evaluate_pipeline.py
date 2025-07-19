import os
import dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from hybrid_retriver import hybrid_retrieve, bm25, collection
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from constant.constants import (
    HF_MODELS,
    OPENAI_MODELS,
    FILE_CANDIDATE_POOL_ADDRESS,
    FILE_DESTINATION_ADDRESS,
)
from constant.climate_framework import climate_assessment_prompt, system_prompt

# Load environment variables
dotenv.load_dotenv()

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ClimateRAG Evaluation CLI")
    parser.add_argument(
        "--model_source", required=False, choices=["hf", "gpt"], help="Model type"
    )
    parser.add_argument(
        "--model_args", type=str, required=False, help="Model name or OpenAI model id"
    )
    parser.add_argument(
        "--tasks", type=str, default="climate", help="Task name (not used here)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=FILE_DESTINATION_ADDRESS,
        help="Path to save output CSV",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=FILE_CANDIDATE_POOL_ADDRESS,
        help="Path to input CSV",
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=5, help="Top-k passages to retrieve"
    )
    parser.add_argument(
        "--max_rows", type=int, default=None, help="Optional: max rows to process"
    )
    return parser.parse_args()


def select_model():
    print("Please select a model to use:")
    print("Hugging Face models:")
    for key, value in HF_MODELS.items():
        print(f"{key}: {value}")

    print("\nOpenAI models:")
    for key, value in OPENAI_MODELS.items():
        print(f"{key}: {value}")

    while True:
        choice = input("\nEnter model number: ")
        if choice in HF_MODELS:
            return HF_MODELS[choice], "hf"
        elif choice in OPENAI_MODELS:
            return OPENAI_MODELS[choice], "gpt"
        else:
            print("Invalid selection, please try again")


def initialize_hf_model(model_name):
    """Initialize Hugging Face model"""
    print(f"Loading Hugging Face model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": 0}, quantization_config=config
    )
    model.gradient_checkpointing_enable()
    return model, tokenizer


def generate_hf_answer(prompt, model, tokenizer):
    """Generate answer using Hugging Face model"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2000)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result[len(prompt) :].strip()


def generate_openai_answer(prompt, model, client):
    """Generate answer using OpenAI model"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Failed to generate answer: {e}")
        return f"Error: {str(e)}"


def generate_answer_from_retrieve(
    query,
    model_name,
    model_type,
    hf_model=None,
    hf_tokenizer=None,
    openai_client=None,
    top_k=5,
):
    """
    Generate IPCC-style structured vulnerability/resilience answer from retrieved document chunks

    Parameters:
        query: Input question
        model_name: Model name
        model_type: Model type ('hf' or 'gpt')
        hf_model: Hugging Face model (if HF model)
        hf_tokenizer: Hugging Face tokenizer
        openai_client: OpenAI client
        top_k: Number of top passages to retrieve

    Returns:
        str: JSON-formatted structured answer
    """
    top_docs = hybrid_retrieve(query, bm25, collection, top_k=top_k)
    context = "\n\n".join([doc for _, _, doc, _, _ in top_docs])

    prompt = climate_assessment_prompt.format(query=query, context=context)

    if model_type == "hf":
        return generate_hf_answer(prompt, hf_model, hf_tokenizer)
    else:
        return generate_openai_answer(prompt, model_name, openai_client)


def process_csv(
    input_file_path,
    output_file_path,
    model_name,
    model_type,
    hf_model=None,
    hf_tokenizer=None,
    openai_client=None,
    max_rows=None,
):
    df = pd.read_csv(input_file_path)

    if max_rows is not None:
        df = df.head(max_rows)
        print(f"[INFO] Processing first {max_rows} rows only")

    queries = []
    generated_jsons = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating evaluations"):
        query = row["query"]
        json_result = generate_answer_from_retrieve(
            query, model_name, model_type, hf_model, hf_tokenizer, openai_client
        )
        queries.append(query)
        generated_jsons.append(json_result)

    results_df = pd.DataFrame(
        {
            "query": queries,
            "generated_structured_answer": generated_jsons,
            "model_used": model_name,
        }
    )

    results_df.to_csv(output_file_path, index=False)
    print(f"\nDone. Results saved to: {output_file_path}")


if __name__ == "__main__":
    args = parse_args()

    if args.model_source == None:
        # Let user select model
        model_name, model_type = select_model()
    else:
        model_name = args.model_args
        model_type = args.model_source
    # Initialize model
    hf_model, hf_tokenizer, openai_client = None, None, None

    if model_type == "hf":
        hf_model, hf_tokenizer = initialize_hf_model(model_name)
    else:
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Process CSV file
    process_csv(
        args.input_path,
        args.output_path,
        model_name,
        model_type,
        hf_model,
        hf_tokenizer,
        openai_client,
        max_rows=100,  # Only process first 100 rows
    )

# Example usage:
# python evaluate.py \
#   --model hf \
#   --model_args eci-io/climategpt-7b \
#   --tasks claim_binary \
#   --input_path ./Ground-truth/QACandidate_Pool.csv \
#   --output_path ./results/climategpt-7b.jsonl \
#   --num_fewshot 5 \
#   --max_rows 100
