import pandas as pd
import openai
from tqdm import tqdm
import time
import os
import dotenv
from openai import OpenAI
from constant.climate_framework import climate_assessment_prompt, generate_ground_truth_with_evidence_prompt

dotenv.load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def generate_answer(query, context):
    prompt = generate_ground_truth_with_evidence_prompt.format(query=query, context=context)
    try:
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
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error: {str(e)}"

def process_csv(input_file_path, output_file_path, max_rows=None):
    df = pd.read_csv(input_file_path)

    if max_rows is not None:
        df = df.head(max_rows)
        print(f"Processing first {max_rows} rows for testing...")

    queries = []
    contexts = []
    generated_answers = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
        query = row['query']
        correct_passage_index = int(row['correct_passage_index'])
        passage_column = f'passage_{correct_passage_index}'
        correct_passage = row[passage_column]
        answer = generate_answer(query, correct_passage)
        queries.append(query)
        contexts.append(correct_passage)
        generated_answers.append(answer)
        time.sleep(0.5)
        
        # Save intermediate results every 10 queries (in case of interruption)
        if (index + 1) % 10 == 0:
            temp_df = pd.DataFrame({
                'query': queries,
                'correct_passage_context': contexts,
                'generated_answer': generated_answers
            })
            temp_df.to_csv(f'{output_file_path}.temp', index=False)
            print(f"Saved intermediate results: {index + 1} queries processed")
    
    results_df = pd.DataFrame({
        'query': queries,
        'correct_passage_context': contexts,
        'generated_answer': generated_answers
    })
    
    # Save to CSV
    results_df.to_csv(output_file_path, index=False)
    
    temp_file = f'{output_file_path}.temp'
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"\nProcessing complete! Results saved to: {output_file_path}")
    return results_df


if __name__ == "__main__":
    # Set your file paths
    input_csv_path = "Ground-truth/QACandidate_Pool.csv"  
    output_csv_path = "Ground-truth/ground_truth.csv"  # Output file
    
    results = process_csv(input_csv_path, output_csv_path, max_rows=100)  

