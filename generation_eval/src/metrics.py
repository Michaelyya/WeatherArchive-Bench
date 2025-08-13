import pandas as pd
import numpy as np
import re
from sklearn.metrics import (
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score,
)
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import bert_score
from constant.constants import FILE_DESTINATION_ADDRESS


def calculate_accuracy(gt_values, model_values):
    if len(gt_values) != len(model_values):
        return 0.0
    
    correct = 0
    total = len(gt_values)
    na_num = 0
    
    print(len(gt_values))
    print(len(model_values))
    
    
    for gt, model in zip(gt_values, model_values):
        if gt == model:  # This works for both integers and strings
            correct += 1
        elif model == "NA":
            na_num += 1
    print("\n correct:", correct)
    print("\n na_total:", na_num)
    return correct / total if total > 0 else 0.0


def extract_first_number(text):
    if pd.isna(text) or text == "":
        return "NA"
    
    text_str = str(text).strip()
    if text_str.lower() in ["nan", "none", "[na]", "na"]:
        return "NA"

    match = re.search(r'\d+', text_str)
    return int(match.group()) if match else None


# def compare_csvs(ground_truth_file, model_output_file):
#     # Load the CSV files
#     gt_df = pd.read_csv(ground_truth_file)
#     model_df = pd.read_csv(model_output_file)
    
#     if len(gt_df) != len(model_df):
#         print(f"Warning: Different number of rows - GT: {len(gt_df)}, Model: {len(model_df)}")
#         return

#     print(f"Comparing {len(gt_df)} rows...")
#     print("=" * 50)

#     # 1. SCORE-BASED EVALUATION
#     print("1. SCORE-BASED METRICS")
#     print("-" * 25)

#     score_columns = ["exposure_score", "sensitivity_score", "adaptability_score"]

#     for col in score_columns:
#         print(f"\nAnalyzing {col}:")

#         # Extract first numbers and convert to integers
#         gt_numbers = [extract_first_number(val) for val in gt_df[col]]
#         model_numbers = [extract_first_number(val) for val in model_df[col]]
        
#         # Filter out "NA" and None values for numeric comparison
#         valid_pairs = [(gt, model) for gt, model in zip(gt_numbers, model_numbers) 
#                        if gt != "NA" and model != "NA" and gt is not None and model is not None]
        
#         if valid_pairs:
#             gt_valid, model_valid = zip(*valid_pairs)
#             mae = mean_absolute_error(gt_valid, model_valid)
#             print(f"  MAE: {mae:.3f} (on {len(valid_pairs)} valid numeric pairs)")
#         else:
#             print(f"  MAE: N/A (no valid numeric values to compare)")
        
#         # Calculate accuracy on all values (including None)
#         accuracy = calculate_accuracy(gt_numbers, model_numbers)
#         print(f"  Accuracy (exact match): {accuracy:.3f}")
#         print(f"  Total samples: {len(gt_numbers)}")
#         print(f"  Valid numeric pairs: {len(valid_pairs)}")

#     # 2. MCQ EVALUATION
#     print("\n2. MCQ METRICS")
#     print("-" * 15)

#     mcq_columns = ["temporal_scale_focus", "functional_system_focus", "spatial_scale_focus"]

#     for col in mcq_columns:
#         print(f"\nAnalyzing {col}:")

#         gt_values = gt_df[col].astype(str)
#         model_values = model_df[col].astype(str)

#         gt_values = gt_values.replace(["nan", "NaN", "None", "[NA]"], "NA")
#         model_values = model_values.replace(["nan", "NaN", "None", "[NA]"], "NA")

#         f1 = f1_score(gt_values, model_values, average="macro", zero_division=0)
#         precision = precision_score(gt_values, model_values, average="macro", zero_division=0)
#         recall = recall_score(gt_values, model_values, average="macro", zero_division=0)

#         print(f"  F1 Score: {f1:.3f}")
#         print(f"  Precision: {precision:.3f}")
#         print(f"  Recall: {recall:.3f}")

#     # 3. QUESTION-ANSWER EVALUATION
#     print("\n3. QUESTION-ANSWER METRICS")
#     print("-" * 27)

#     qa_column = "answer"
#     gt_answers = gt_df[qa_column].astype(str)
#     model_answers = model_df[qa_column].astype(str)

#     # Calculate BLEU scores
#     bleu_scores = []
#     for gt_answer, model_answer in zip(gt_answers, model_answers):
#         if gt_answer == "nan" or model_answer == "nan":
#             bleu_scores.append(1.0 if gt_answer == model_answer else 0.0)
#         elif gt_answer.strip() == "" or model_answer.strip() == "":
#             bleu_scores.append(0.0)
#         else:
#             gt_words = gt_answer.split()
#             model_words = model_answer.split()
#             bleu = sentence_bleu([gt_words], model_words) if model_words else 0.0
#             bleu_scores.append(bleu)

#     avg_bleu = np.mean(bleu_scores)
    
#     # Calculate ROUGE scores
#     rouge = Rouge()
#     rouge_1_scores = []
#     rouge_l_scores = []

#     for gt_answer, model_answer in zip(gt_answers, model_answers):
#         if gt_answer == "nan" or model_answer == "nan":
#             score = 1.0 if gt_answer == model_answer else 0.0
#             rouge_1_scores.append(score)
#             rouge_l_scores.append(score)
#         elif gt_answer.strip() == "" or model_answer.strip() == "":
#             rouge_1_scores.append(0.0)
#             rouge_l_scores.append(0.0)
#         else:
#             scores = rouge.get_scores(model_answer, gt_answer)
#             rouge_1_scores.append(scores[0]["rouge-1"]["f"])
#             rouge_l_scores.append(scores[0]["rouge-l"]["f"])

#     avg_rouge_1 = np.mean(rouge_1_scores)
#     avg_rouge_l = np.mean(rouge_l_scores)

#     # Calculate BERTScore
#     P, R, F1 = bert_score.score(model_answers.tolist(), gt_answers.tolist(), lang="en", verbose=False)
#     avg_bert_f1 = F1.mean().item()

#     print(f"\nFinal Results:")
#     print(f"BLEU Score: {avg_bleu:.3f}")
#     print(f"ROUGE-1: {avg_rouge_1:.3f}")
#     print(f"ROUGE-L: {avg_rouge_l:.3f}")
#     print(f"BERTScore F1: {avg_bert_f1:.3f}")

    # return avg_bleu, avg_rouge_1, avg_rouge_l, avg_bert_f1


def evaluate_single_model(ground_truth_file, model_file):
    gt_df = pd.read_csv(ground_truth_file)
    model_df = pd.read_csv(model_file)
    
    
    results = {}

    # 1. SCORE-BASED EVALUATION
    score_columns = ["exposure_score", "sensitivity_score", "adaptability_score"]
    for col in score_columns:
        # Extract first numbers and convert to integers
        gt_numbers = [extract_first_number(val) for val in gt_df[col]]
        model_numbers = [extract_first_number(val) for val in model_df[col]]
        
        # Calculate accuracy on all values (including None)
        accuracy = calculate_accuracy(gt_numbers, model_numbers)
        results[f"{col}_accuracy"] = accuracy
        
        # Filter out "NA" and None values for numeric comparison
        valid_pairs = [(gt, model) for gt, model in zip(gt_numbers, model_numbers) 
                       if gt != "NA" and model != "NA" and gt is not None and model is not None]
        
        if valid_pairs:
            gt_valid, model_valid = zip(*valid_pairs)
            results[f"{col}_mae"] = mean_absolute_error(gt_valid, model_valid)
        else:
            results[f"{col}_mae"] = float("nan")

    # 2. MCQ EVALUATION
    mcq_columns = [
        "temporal_scale_focus",
        "functional_system_focus",
        "spatial_scale_focus",
    ]
    for col in mcq_columns:
        gt_values = gt_df[col].astype(str).replace(["nan", "NaN", "None", "[NA]"], "NA")
        model_values = (
            model_df[col].astype(str).replace(["nan", "NaN", "None", "[NA]"], "NA")
        )

        results[f"{col}_f1"] = f1_score(
            gt_values, model_values, average="macro", zero_division=0
        )
        results[f"{col}_precision"] = precision_score(
            gt_values, model_values, average="macro", zero_division=0
        )
        results[f"{col}_recall"] = recall_score(
            gt_values, model_values, average="macro", zero_division=0
        )

    # 3. QUESTION-ANSWER EVALUATION
    qa_column = "answer"
    gt_answers = gt_df[qa_column].astype(str)
    model_answers = model_df[qa_column].astype(str)

    # BLEU scores
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

    # ROUGE scores
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

    # BERTScore
    P, R, F1 = bert_score.score(
        model_answers.tolist(), gt_answers.tolist(), lang="en", verbose=False
    )
    results["bertscore_f1"] = F1.mean().item()

    return results


if __name__ == "__main__":
    import os
    import glob
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs against ground truth"
    )
    parser.add_argument(
        "--ground-truth",
        "-gt",
        default="ground-truth/ground_truth.csv",
        help="Path to ground truth CSV file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="generation_eval/output",
        help="Directory containing model output CSV files",
    )
    parser.add_argument(
        "--output-file",
        "-f",
        default="model_evaluation_summary.csv",
        help="Output CSV file for summary results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output showing detailed evaluation for each model",
    )

    args = parser.parse_args()

    # Get all model result files
    model_files = glob.glob(os.path.join(args.output_dir, "*.csv"))
    print("model_files:", model_files)

    print("=" * 80)
    print("EVALUATING ALL MODELS AGAINST GROUND TRUTH")
    print("=" * 80)
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Found {len(model_files)} model result files")
    print("=" * 80)

    # Store all results for summary table
    all_results = []

    # Evaluate each model
    for model_file in sorted(model_files):
        model_name = os.path.basename(model_file).replace("-results.csv", "")
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")

        if args.verbose:
            # Run detailed evaluation with print output
            compare_csvs(args.ground_truth, model_file)
        else:
            # Run evaluation and collect metrics
            results = evaluate_single_model(args.ground_truth, model_file)
            if results:
                results["model_name"] = model_name
                all_results.append(results)
                print(f"Completed evaluation of {model_name}")
            else:
                print(f"Failed to evaluate {model_name}")

    # Create summary table
    if all_results:
        print(f"\n{'='*80}")
        print("CREATING SUMMARY TABLE")
        print(f"{'='*80}")

        # Convert to DataFrame and organize
        summary_df = pd.DataFrame(all_results)
        cols = ["model_name"] + [
            col for col in summary_df.columns if col != "model_name"
        ]
        summary_df = summary_df[cols]

        # Round numeric columns
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        summary_df[numeric_cols] = summary_df[numeric_cols].round(3)

        # Save and display
        summary_df.to_csv(args.output_file, index=False)
        print(f"Summary saved to: {args.output_file}")

        print("\nSUMMARY TABLE:")
        print("=" * 120)
        print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("ALL MODEL EVALUATIONS COMPLETED")
    print(f"{'='*80}")
