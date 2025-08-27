import pandas as pd
import numpy as np
import re
from sklearn.metrics import (
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import bert_score
import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)


def quadratic_weighted_kappa(y_true, y_pred, min_rating=1, max_rating=5):
    valid_mask = [(gt != "NA" and gt is not None and pred != "NA" and pred is not None) 
                   for gt, pred in zip(y_true, y_pred)]
    
    if not any(valid_mask):
        return float('nan')
    
    y_true_valid = [y_true[i] for i in range(len(y_true)) if valid_mask[i]]
    y_pred_valid = [y_pred[i] for i in range(len(y_pred)) if valid_mask[i]]
    
    if len(y_true_valid) == 0:
        return float('nan')
    
    y_true_valid = [int(gt) for gt in y_true_valid]
    y_pred_valid = [int(pred) for pred in y_pred_valid]
    
    kappa = cohen_kappa_score(y_true_valid, y_pred_valid, 
                             labels=list(range(min_rating, max_rating + 1)),
                             weights='quadratic')
    
    return kappa


def llm_judge_answer(oracle_answer, generated_answer, model_name="gpt-4.1"):
    prompt = f"""You are an expert evaluator. Compare the oracle answer with the generated answer and determine if the generated answer COVERS the key information stated in the oracle answer.

Oracle Answer: {oracle_answer}

Generated Answer: {generated_answer}

Task: Determine if the generated answer COVERS the key information from the oracle answer.

Consider:
- Does the generated answer contain the main points from the oracle answer?
- Is the information accurate and relevant?
- Does it address the same question/topic?

Output ONLY: "true" if it covers, "false" if it doesn't cover.

Your response:"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=50
    )
    
    judge_output = response.choices[0].message.content.strip().lower()

    if "true" in judge_output:
        return 1
    else:
        return 0


def extract_first_number(text):
    if pd.isna(text) or text == "":
        return "NA"
    
    text_str = str(text).strip()
    if text_str.lower() in ["nan", "none", "[na]", "na"]:
        return "NA"
    
    match = re.search(r'\d+', text_str)
    return int(match.group()) if match else None


def evaluate_single_model(ground_truth_file, model_file):
    gt_df = pd.read_csv(ground_truth_file)
    model_df = pd.read_csv(model_file)
    
    results = {}
    
    score_columns = ["exposure_score", "sensitivity_score", "adaptability_score"]
    for col in score_columns:
        gt_numbers = [extract_first_number(val) for val in gt_df[col]]
        model_numbers = [extract_first_number(val) for val in model_df[col]]
        
        qwk_score = quadratic_weighted_kappa(gt_numbers, model_numbers)
        results[f"{col}_qwk"] = qwk_score
        
        valid_pairs = [(gt, model) for gt, model in zip(gt_numbers, model_numbers) 
                       if gt != "NA" and model != "NA" and gt is not None and model is not None]
        
        if valid_pairs:
            gt_valid, model_valid = zip(*valid_pairs)
            results[f"{col}_mae"] = mean_absolute_error(gt_valid, model_valid)
        else:
            results[f"{col}_mae"] = float("nan")
    
    mcq_columns = ["temporal_scale_focus", "functional_system_focus", "spatial_scale_focus"]
    for col in mcq_columns:
        gt_values = gt_df[col].astype(str).replace(["nan", "NaN", "None", "[NA]"], "NA")
        model_values = model_df[col].astype(str).replace(["nan", "NaN", "None", "[NA]"], "NA")
        
        results[f"{col}_f1"] = f1_score(gt_values, model_values, average="macro", zero_division=0)
        results[f"{col}_precision"] = precision_score(gt_values, model_values, average="macro", zero_division=0)
        results[f"{col}_recall"] = recall_score(gt_values, model_values, average="macro", zero_division=0)
    
    qa_column = "answer"
    gt_answers = gt_df[qa_column].astype(str)
    model_answers = model_df[qa_column].astype(str)
    
    bleu_scores = []
    for gt_answer, model_answer in zip(gt_answers, model_answers):
        if gt_answer == "nan" or model_answer == "nan":
            bleu_scores.append(1.0 if gt_answer == model_answer else 0.0)
        elif gt_answer.strip() == "" or model_answer.strip() == "":
            bleu_scores.append(0.0)
        else:
            gt_words = gt_answer.split()
            model_words = model_answer.split()
            bleu = sentence_bleu([gt_words], model_words) if model_words else 0.0
            bleu_scores.append(bleu)
    
    results["bleu_score"] = np.mean(bleu_scores)
    
    rouge = Rouge()
    rouge_1_scores = []
    rouge_l_scores = []
    
    for gt_answer, model_answer in zip(gt_answers, model_answers):
        if gt_answer == "nan" or model_answer == "nan":
            score = 1.0 if gt_answer == model_answer else 0.0
            rouge_1_scores.append(score)
            rouge_l_scores.append(score)
        elif gt_answer.strip() == "" or model_answer.strip() == "":
            rouge_1_scores.append(0.0)
            rouge_l_scores.append(0.0)
        else:
            scores = rouge.get_scores(model_answer, gt_answer)
            rouge_1_scores.append(scores[0]["rouge-1"]["f"])
            rouge_l_scores.append(scores[0]["rouge-l"]["f"])
    
    results["rouge_1"] = np.mean(rouge_1_scores)
    results["rouge_l"] = np.mean(rouge_l_scores)
    
    P, R, F1 = bert_score.score(model_answers.tolist(), gt_answers.tolist(), lang="en", verbose=False)
    results["bertscore_f1"] = F1.mean().item()
    
    print("  Running LLM-as-a-Judge evaluation...")
    llm_judge_scores = []
    for gt_answer, model_answer in zip(gt_answers, model_answers):
        if gt_answer == "nan" or model_answer == "nan":
            llm_judge_scores.append(1.0 if gt_answer == model_answer else 0.0)
        elif gt_answer.strip() == "" or model_answer.strip() == "":
            llm_judge_scores.append(0.0)
        else:
            judge_score = llm_judge_answer(gt_answer, model_answer)
            llm_judge_scores.append(judge_score)
    
    results["llm_judge_score"] = np.mean(llm_judge_scores)
    
    return results


if __name__ == "__main__":
    import os
    import glob
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model outputs against ground truth")
    parser.add_argument("--ground-truth", "-gt", default="ground-truth/ground_truth.csv", 
                       help="Path to ground truth CSV file")
    parser.add_argument("--output-dir", "-o", default="generation_eval/output", 
                       help="Directory containing model output CSV files")
    parser.add_argument("--output-file", "-f", default="model_evaluation_summary.csv",
                       help="Output CSV file for summary results")
    
    args = parser.parse_args()
    
    model_files = glob.glob(os.path.join(args.output_dir, "*.csv"))
    
    print("=" * 80)
    print("EVALUATING ALL MODELS AGAINST GROUND TRUTH")
    print("=" * 80)
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Found {len(model_files)} model result files")
    print("=" * 80)
    
    all_results = []
    for model_file in sorted(model_files):
        model_name = os.path.basename(model_file).replace("-results.csv", "")
        print(f"Evaluating: {model_name}")
        
        results = evaluate_single_model(args.ground_truth, model_file)
        if results:
            results["model_name"] = model_name
            all_results.append(results)
            print(f"  ✓ Completed")
        else:
            print(f"  ✗ Failed")
    
    if all_results:
        print(f"\n{'='*80}")
        print("CREATING SUMMARY TABLE")
        print("=" * 80)
        
        summary_df = pd.DataFrame(all_results)
        cols = ["model_name"] + [col for col in summary_df.columns if col != "model_name"]
        summary_df = summary_df[cols]
        
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        summary_df[numeric_cols] = summary_df[numeric_cols].round(3)
        
        summary_df.to_csv(args.output_file, index=False)
        print(f"Summary saved to: {args.output_file}")
        
        print("\nSUMMARY TABLE:")
        print("=" * 120)
        print(summary_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("ALL MODEL EVALUATIONS COMPLETED")
    print(f"{'='*80}")
