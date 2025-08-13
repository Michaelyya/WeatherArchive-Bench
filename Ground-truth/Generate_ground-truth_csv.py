import pandas as pd
import openai
from tqdm import tqdm
import time
import os
import dotenv
import json
from openai import OpenAI
from constant.climate_framework import climate_assessment_prompt

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def generate_answer(query, context):
    prompt = climate_assessment_prompt.format(query=query, context=context)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a climate expert who creates structured vulnerability and resilience assessments following IPCC frameworks."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content.strip()
    

def parse_response(response):
    lines = response.strip().split('\n')
    parsed_data = {}
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            # Map to our CSV column names
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
    
    # Fill in any missing keys with default values
    default_keys = [
        'region', 'exposure_score', 'sensitivity_score', 'adaptability_score', 
        'temporal_scale_focus', 'functional_system_focus',
        'spatial_scale_focus', 'answer'
    ]
    
    for key in default_keys:
        if key not in parsed_data:
            parsed_data[key] = 'NA'
    
    # parsed_data['full_response'] = response
    return parsed_data
        
def process_csv(input_file_path, output_file_path, max_rows=None):
    df = pd.read_csv(input_file_path)

    if max_rows is not None:
        df = df.head(max_rows)
        print(f"Processing first {max_rows} rows for testing...")

    # Initialize lists for all columns
    queries = []
    contexts = []
    regions = []
    exposure_scores = []
    sensitivity_scores = []
    adaptability_scores = []
    temporal_scale_focuses = []
    functional_system_focuses = []
    spatial_scale_focuses = []
    answers = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
        query = row['query']
        correct_passage_index = int(row['correct_passage_index'])
        passage_column = f'passage_{correct_passage_index}'
        correct_passage = row[passage_column]
        
        # Generate response
        response = generate_answer(query, correct_passage)
        
        # Parse response into individual components
        parsed_data = parse_response(response)
        
        # Append to lists
        queries.append(query)
        contexts.append(correct_passage)
        regions.append(parsed_data['region'])
        exposure_scores.append(parsed_data['exposure_score'])
        sensitivity_scores.append(parsed_data['sensitivity_score'])
        adaptability_scores.append(parsed_data['adaptability_score'])
        temporal_scale_focuses.append(parsed_data['temporal_scale_focus'])
        functional_system_focuses.append(parsed_data['functional_system_focus'])
        spatial_scale_focuses.append(parsed_data['spatial_scale_focus'])
        answers.append(parsed_data['answer'])
        
        time.sleep(0.5)
        
        # Save intermediate results every 10 queries
        if (index + 1) % 10 == 0:
            temp_df = pd.DataFrame({
                'query': queries,
                'correct_passage_context': contexts,
                'region': regions,
                'exposure_score': exposure_scores,
                'sensitivity_score': sensitivity_scores,
                'adaptability_score': adaptability_scores,
                'temporal_scale_focus': temporal_scale_focuses,
                'functional_system_focus': functional_system_focuses,
                'spatial_scale_focus': spatial_scale_focuses,
                'answer': answers,
            })
            temp_df.to_csv(f'{output_file_path}.temp', index=False)
            print(f"Saved intermediate results: {index + 1} queries processed")
    
    # Create final DataFrame with all columns
    results_df = pd.DataFrame({
        'query': queries,
        'correct_passage_context': contexts,
        'region': regions,
        'exposure_score': exposure_scores,
        'sensitivity_score': sensitivity_scores,
        'adaptability_score': adaptability_scores,
        'temporal_scale_focus': temporal_scale_focuses,
        'functional_system_focus': functional_system_focuses,
        'spatial_scale_focus': spatial_scale_focuses,
        'answer': answers
    })
    results_df.to_csv(output_file_path, index=False)
    
    # Remove temp file
    temp_file = f'{output_file_path}.temp'
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return results_df


if __name__ == "__main__":
    input_csv_path = "Ground-truth/QACandidate_Pool.csv"  
    output_csv_path = "Ground-truth/ground_truth.csv" 
    
    results = process_csv(input_csv_path, output_csv_path, max_rows=335)