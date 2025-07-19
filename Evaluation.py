import os
import dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from WXImpactRAG.hybrid_retriver import hybrid_retrieve, bm25_model, collection
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re

# Load environment variables
dotenv.load_dotenv()

# Available Hugging Face models
HF_MODELS = {
    "1": "meta-llama/Meta-Llama-3-8B-Instruct",
    "2": "Qwen/Qwen2.5-7B-Instruct",
    "3": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "4": "Qwen/Qwen2.5-14B-Instruct",
    "5": "google/gemma-2-9b-it",
    "6": "mistralai/Mistral-Small-24B-Instruct-2501",
}

# config.ymal

OPENAI_MODELS = {"7": "gpt-3.5-turbo", "8": "gpt-4"}


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
                    "content": "You are a climate expert who creates structured vulnerability and resilience assessments following IPCC frameworks. Generate evidence-based JSON responses using provided document chunks.",
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
    top_docs = hybrid_retrieve(query, bm25_model, collection, top_k=top_k)
    context = "\n\n".join([doc for _, _, doc, _, _ in top_docs])

    prompt = f"""You are a climate vulnerability and resilience expert. Create a comprehensive assessment following IPCC vulnerability framework and multi-scale resilience analysis.

VULNERABILITY FRAMEWORK:
- Exposure: Assess the degree of climate stress (long-term climate changes, variability, and extreme event magnitude/frequency)
- Sensitivity: Evaluate how the system responds to climate change (precondition for vulnerability - higher sensitivity = greater potential impacts)  
- Adaptability: Determine capacity to adjust to climate stimuli based on wealth, technology, education, information, skills, infrastructure, resources, and governance
NOTE: If no evidence is found, output a score of "NA" and "No evidence found." in evidence.

RESILIENCE FRAMEWORK:
- Temporal Scale: Choose primary focus among short-term absorptive capacity (emergency responses), medium-term adaptive capacity (policy/infrastructure adjustments), or long-term transformative capacity (systemic redesign/migration)
- Functional System Scale: Select 1-3 key affected systems from health, energy, food, water, transportation, information. Consider redundancy, robustness, recovery time, interdependence
- Spatial Scale: Choose primary level among local, community, regional, national. Highlight capacity differences across scales
NOTE: If no evidence is found, output a primary_focus of "NA" and "No evidence found." in evidence.

INSTRUCTIONS:
- Use 1-5 scale (1=very low, 5=very high) with evidence from document chunks
- Quote directly from chunks when possible, clearly indicate paraphrasing
- Ensure all scores and selections are supported by evidence

INPUT:
Query: {query}

Retrieved Document Chunk:
{context}

OUTPUT FORMAT (JSON):
{{
  "region": "[Extract/infer geographic region]",
  "vulnerability": {{
    "exposure": {{
      "score": [1-5],
      "evidence": "Direct quotes/paraphrases supporting climate stress assessment"
    }},
    "sensitivity": {{
      "score": [1-5], 
      "evidence": "Direct quotes/paraphrases supporting system response assessment"
    }},
    "adaptability": {{
      "score": [1-5],
      "evidence": "Direct quotes/paraphrases supporting adaptive capacity assessment"
    }}
  }},
  "resilience": {{
    "temporal_scale": {{
      "primary_focus": "[short-term absorptive capacity | medium-term adaptive capacity | long-term transformative capacity]",
      "evidence": "Supporting evidence from chunks"
    }},
    "functional_system": {{
      "primary_focus": ["[1-3 from: health, energy, food, water, transportation, information]"], 
      "evidence": "Supporting evidence from chunks"
    }},
    "spatial_scale": {{
      "primary_focus": "[local | community | regional | national]",
      "evidence": "Supporting evidence from chunks"
    }}
  }},
  "question_answer": {{
    "question": "{query}",
    "answer": "[2-3 sentence concise answer addressing query based on chunks]"
  }}
}}

Only output the JSON response. Do not include any additional text and space in the response."""

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
    # Let user select model
    model_name, model_type = select_model()

    # Initialize model
    hf_model, hf_tokenizer, openai_client = None, None, None

    if model_type == "hf":
        hf_model, hf_tokenizer = initialize_hf_model(model_name)
    else:
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Set file paths
    input_csv_path = (
        "C:/Users/14821/PyCharmMiscProject/Ground-truth/QACandidate_Pool.csv"
    )
    output_csv_path = "C:/Users/14821/PyCharmMiscProject/Ground-truth/generated_structured_answers.csv"

    # Process CSV file
    process_csv(
        input_csv_path,
        output_csv_path,
        model_name,
        model_type,
        hf_model,
        hf_tokenizer,
        openai_client,
        max_rows=100,  # Only process first 100 rows
    )
