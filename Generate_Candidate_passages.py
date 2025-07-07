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
    prompt = f"""You are a meteological assistant that answers questions based on the provided context.

    Please include the following key factors in your answer with regard to the vulerability of the disruptive weather: 
    - Exposure – the degree of climate stress upon a particular unit of analysis. Climate stress can refer to long-term changes in climate conditions or to changes in climate variability and the magnitude and frequency of extreme events.
    - Sensitivity – the degree to which a system will respond, either positively or negatively, to a change in climate. Climate sensitivity can be considered a precondition for vulnerability: the more sensitive an exposure unit is to climate change, the greater are the potential impacts, and hence the more vulnerable.
    - Adaptability – the capacity of a system to adjust in response to actual or expected climate stimuli, their effects, or impacts. The latest IPCC report identifies adaptive capacity as 'a function of wealth, technology, education, information, skills, infrastructure, access to resources, and stability and management capabilities'.

    Support resilience claims with specific evidence of:
    - Economic indicators (income, employment diversity, resources)
    - Social factors (education, infrastructure, institutions)
    - Environmental conditions (exposure to climate hazards)
    - Adaptive capacity indicators (technology, governance, flexibility)

    Question: {query}

    Context: {context}

    Please provide a comprehensive answer to the question based ONLY on the information provided in the context. If the context doesn't contain enough information to answer the question, state that clearly. 
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions accurately based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
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
    input_csv_path = "QACandidate_Pool.csv"  
    output_csv_path = "qa_results_with_answers.csv"  # Output file
    
    results = process_csv(input_csv_path, output_csv_path, max_rows=2)  # Process only 2 rows

    
    print("\nFirst 3 results:")
    for i in range(min(3, len(results))):
        print(f"\n--- Query {i+1} ---")
        print(f"Question: {results.iloc[i]['query']}")
        print(f"Answer: {results.iloc[i]['generated_answer'][:200]}...")  