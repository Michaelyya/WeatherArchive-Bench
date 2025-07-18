import pandas as pd
import openai
from tqdm import tqdm
import time
import os
import dotenv
import json
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def generate_answer(query, context):
    prompt = f"""You are a climate vulnerability and resilience expert. Create a comprehensive assessment following IPCC vulnerability framework and multi-scale resilience analysis.

VULNERABILITY FRAMEWORK:
- **Exposure**: Assess the degree of climate stress (long-term climate changes, variability, and extreme event magnitude/frequency)
- **Sensitivity**: Evaluate how the system responds to climate change (precondition for vulnerability - higher sensitivity = greater potential impacts)   
- **Adaptability**: Determine capacity to adjust to climate stimuli based on wealth, technology, education, information, skills, infrastructure, resources, and governance

RESILIENCE FRAMEWORK:
- **Temporal Scale**: Choose primary focus among short-term absorptive capacity (emergency responses), medium-term adaptive capacity (policy/infrastructure adjustments), or long-term transformative capacity (systemic redesign/migration)
- **Functional System Scale**: Select 1-3 key affected systems from health, energy, food, water, transportation, information. Consider redundancy, robustness, recovery time, interdependence
- **Spatial Scale**: Choose primary level among local, community, regional, national. Highlight capacity differences across scales

INSTRUCTIONS:
- Use 1-5 scale (1=very low, 5=very high) with evidence from document chunks
- Quote directly from chunks when possible, clearly indicate paraphrasing
- Ensure all scores and selections are supported by evidence, only output as instructed

INPUT:
Query: {query}
Retrieved Document Chunk: {context}

OUTPUT FORMAT (follow this exact structure):
Region: [Extract/infer geographic region]
Exposure: [1-5 or NA]
Sensitivity: [1-5 or NA]
Adaptability: [1-5 or NA]
Temporal_Scale: [short-term absorptive capacity | medium-term adaptive capacity | long-term transformative capacity | NA]
Functional_System: [health, energy, food, water, transportation, information - list 1-3 separated by commas | NA]
Spatial_Scale: [local | community | regional | national | NA]
Answer: [2-3 sentence concise answer addressing query based on chunks]

Only output in the exact format above. Do not include any additional text.   
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a climate expert who creates structured vulnerability and resilience assessments following IPCC frameworks. Generate evidence-based JSON responses using provided document chunks."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000
    )
    
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
    
    parsed_data['full_response'] = response
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
    
    # Save to CSV
    results_df.to_csv(output_file_path, index=False)
    
    # Remove temp file
    temp_file = f'{output_file_path}.temp'
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"\nProcessing complete! Results saved to: {output_file_path}")
    print(f"CSV contains {len(results_df)} rows with the following columns:")
    print(list(results_df.columns))
    
    return results_df


if __name__ == "__main__":
    input_csv_path = "Ground-truth/QACandidate_Pool.csv"  
    output_csv_path = "Ground-truth/ground_truth_csv.csv" 
    
    results = process_csv(input_csv_path, output_csv_path, max_rows=100)