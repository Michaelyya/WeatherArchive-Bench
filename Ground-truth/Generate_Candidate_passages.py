import pandas as pd
import openai
from tqdm import tqdm
import time
import os
import dotenv
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
        NOTE: If no evidence is found, output a score of "NA" and "No evidence found." in evidence.

        RESILIENCE FRAMEWORK:
        - **Temporal Scale**: Choose primary focus among short-term absorptive capacity (emergency responses), medium-term adaptive capacity (policy/infrastructure adjustments), or long-term transformative capacity (systemic redesign/migration)
        - **Functional System Scale**: Select 1-3 key affected systems from health, energy, food, water, transportation, information. Consider redundancy, robustness, recovery time, interdependence
        - **Spatial Scale**: Choose primary level among local, community, regional, national. Highlight capacity differences across scales
        NOTE: If no evidence is found, output a primary_focus of "NA" and "No evidence found." in evidence.
    
        INSTRUCTIONS:
        - Use 1-5 scale (1=very low, 5=very high) with evidence from document chunks
        - Quote directly from chunks when possible, clearly indicate paraphrasing
        - Ensure all scores and selections are supported by evidence

        INPUT:
        Query: {query}
        Retrieved Document Chunk: {context}

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
            "question": {query},
            "answer": "[2-3 sentence concise answer addressing query based on chunks]"
        }}
}}
     
    Only output the JSON response. Do not include any additional text and space in the response."""

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

