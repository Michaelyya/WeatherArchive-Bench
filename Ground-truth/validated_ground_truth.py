import pandas as pd
import openai
import time
import json
import os
from typing import Dict, Any
import os
import dotenv
from openai import OpenAI
from constant.climate_framework import validate_ground_truth

dotenv.load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def validate_and_update_answer_with_gpt4o(query: str, passage: str, generated_answer: str) -> str:
    prompt = validate_ground_truth.format(query=query, passage=passage, generated_answer=generated_answer)
    
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
    
    output_df['score_updated'] = False
    output_df['processing_notes'] = ''
    
    for index, row in output_df.iterrows():
        query = row['query']
        passage = row['correct_passage_context']
        generated_answer = row['generated_answer']
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
    log_file = output_file.replace('.csv', '_log.csv')
    output_df.to_csv(log_file, index=False)
    
    total_processed = len(output_df)
    total_updated = len(output_df[output_df['score_updated'] == True])
    
    print(f"\nSummary:")
    print(f"Total rows processed: {total_processed}")
    print(f"Total rows with score updates: {total_updated}")

if __name__ == "__main__":
    input_file = "Ground-truth/ground_truth.csv"
    output_file = "ground_truth_validated.csv"
    process_csv(input_file, output_file)