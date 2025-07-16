import pandas as pd
import openai
import time
import json
import os
from typing import Dict, Any
import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def validate_and_update_answer_with_gpt4o(query: str, passage: str, generated_answer: str) -> str:
    prompt = f"""
    You are an expert evaluator tasked with validating and updating vulnerability and resilience assessments based on provided context.

    Query: {query}
    
    Context/Passage: {passage}
    
    Current Generated Answer: {generated_answer}
    
    Please carefully review the current answer and:
    1. Validate if the vulnerability scores (exposure, sensitivity, adaptability) are appropriate based on the context
    2. Validate if the resilience assessments are accurate
    3. Update any scores that seem inappropriate or inaccurate
    4. Please be harsh and critical in your evaluation as a human judger
    
    Return the corrected answer in the EXACT same JSON format as provided, but with updated scores if needed:
    
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
    
    Important: Return ONLY the JSON, no additional text or explanations.
    NOTE: Please be harsh and critical in your evaluation as a human judger!
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a climate expert who creates structured vulnerability and resilience assessments following IPCC frameworks. Generate evidence-based JSON responses using provided document chunks."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000
    )
    
    return response.choices[0].message.content.strip()


def has_scores_changed(original_answer: str, updated_answer: str) -> bool:
    try:
        orig_json = json.loads(original_answer)
        updated_json = json.loads(updated_answer)
        
        orig_exposure = orig_json.get('vulnerability', {}).get('exposure', {}).get('score')
        orig_sensitivity = orig_json.get('vulnerability', {}).get('sensitivity', {}).get('score')
        orig_adaptability = orig_json.get('vulnerability', {}).get('adaptability', {}).get('score')
        
        updated_exposure = updated_json.get('vulnerability', {}).get('exposure', {}).get('score')
        updated_sensitivity = updated_json.get('vulnerability', {}).get('sensitivity', {}).get('score')
        updated_adaptability = updated_json.get('vulnerability', {}).get('adaptability', {}).get('score')
        
        return (orig_exposure != updated_exposure or 
                orig_sensitivity != updated_sensitivity or 
                orig_adaptability != updated_adaptability)
        
    except (json.JSONDecodeError, KeyError):
        return False

def process_csv(input_file: str, output_file: str):

    df = pd.read_csv(input_file)
    
    expected_columns = ['query', 'correct_passage_context', 'generated_answer']
    
    output_df = pd.DataFrame({
        'query': df['query'],
        'correct_passage_context': df['correct_passage_context'],
        'generated_answer': df['generated_answer'].copy()
    })
    
    # Add tracking columns
    output_df['score_updated'] = False
    output_df['processing_notes'] = ''
    
    print(f"Processing {len(output_df)} rows...")
    
    for index, row in output_df.iterrows():
        print(f"Processing row {index + 1}/{len(output_df)}...")
        
        query = row['query']
        passage = row['correct_passage_context']
        generated_answer = row['generated_answer']

        print(f"  Row {index + 1}: Validating with GPT-4o...")
        updated_answer = validate_and_update_answer_with_gpt4o(query, passage, generated_answer)
        
        if has_scores_changed(generated_answer, updated_answer):
            output_df.at[index, 'generated_answer'] = updated_answer
            output_df.at[index, 'score_updated'] = True
            output_df.at[index, 'processing_notes'] = 'Scores updated by GPT-4o'
            print(f"  Row {index + 1}: Scores updated")
        else:
            output_df.at[index, 'processing_notes'] = 'Validated - no changes needed'
            print(f"  Row {index + 1}: No score changes needed")

    final_output = output_df[['query', 'correct_passage_context', 'generated_answer']].copy()
    
    final_output.to_csv(output_file, index=False)
    print(f"\nProcessing complete! Updated CSV saved to: {output_file}")
    
    log_file = output_file.replace('.csv', '_log.csv')
    output_df.to_csv(log_file, index=False)
    print(f"Detailed processing log saved to: {log_file}")
    
    total_processed = len(output_df)
    total_updated = len(output_df[output_df['score_updated'] == True])
    
    print(f"\nSummary:")
    print(f"Total rows processed: {total_processed}")
    print(f"Total rows with score updates: {total_updated}")

if __name__ == "__main__":
    input_file = "Ground-truth/ground_truth.csv"
    output_file = "ground_truth_validated.csv"
    process_csv(input_file, output_file)