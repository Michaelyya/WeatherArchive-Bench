import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score,
)
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import bert_score
from constant.constants import FILE_DESTINATION_ADDRESS


def compare_csvs(ground_truth_file, model_output_file):
    # Load the CSV files
    gt_df = pd.read_csv(ground_truth_file)
    model_df = pd.read_csv(model_output_file)
    if len(gt_df) != len(model_df):
        print(
            f"Warning: Different number of rows - GT: {len(gt_df)}, Model: {len(model_df)}"
        )
        return

    print(f"Comparing {len(gt_df)} rows...")
    print("=" * 50)

    # 1. SCORE-BASED EVALUATION (first 3 columns)
    print("1. SCORE-BASED METRICS")
    print("-" * 25)

    score_columns = ["exposure_score", "sensitivity_score", "adaptability_score"]

    for col in score_columns:
        print(f"\nAnalyzing {col}:")

        gt_values = gt_df[col].astype(str)
        model_values = model_df[col].astype(str)

        gt_values = gt_values.replace(["nan", "NaN", "None", "[NA]"], "NA")
        model_values = model_values.replace(["nan", "NaN", "None", "[NA]"], "NA")

        accuracy = accuracy_score(gt_values, model_values)
        gt_numeric = pd.to_numeric(gt_df[col], errors="coerce")
        model_numeric = pd.to_numeric(model_df[col], errors="coerce")

        # Count how many are actually numeric
        gt_numeric_count = gt_numeric.notna().sum()
        model_numeric_count = model_numeric.notna().sum()
        print(f"  GT numeric values: {gt_numeric_count}/{len(gt_df)}")
        print(f"  Model numeric values: {model_numeric_count}/{len(model_df)}")

        # Only consider numeric values
        numeric_mask = ~(gt_numeric.isna() | model_numeric.isna())
        gt_numeric_valid = gt_numeric[numeric_mask]
        model_numeric_valid = model_numeric[numeric_mask]

        if len(gt_numeric_valid) > 0:
            mae = mean_absolute_error(gt_numeric_valid, model_numeric_valid)
            print(f"  MAE: {mae:.3f} (on {len(gt_numeric_valid)} numeric pairs)")
        else:
            print(f"  MAE: N/A (no numeric values to compare)")

        print(f"  Accuracy (exact match): {accuracy:.3f}")
        print(f"  Total samples: {len(gt_values)}")

    # 2. MCQ EVALUATION
    print("\n2. MCQ METRICS")
    print("-" * 15)

    mcq_columns = [
        "temporal_scale_focus",
        "functional_system_focus",
        "spatial_scale_focus",
    ]

    for col in mcq_columns:
        print(f"\nAnalyzing {col}:")

        gt_values = gt_df[col].astype(str)
        model_values = model_df[col].astype(str)

        gt_values = gt_values.replace(["nan", "NaN", "None", "[NA]"], "NA")
        model_values = model_values.replace(["nan", "NaN", "None", "[NA]"], "NA")

        # F1, Precision, Recall
        f1 = f1_score(gt_values, model_values, average="macro", zero_division=0)
        precision = precision_score(
            gt_values, model_values, average="macro", zero_division=0
        )
        recall = recall_score(gt_values, model_values, average="macro", zero_division=0)

        print(f"  F1 Score: {f1:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")

    # 3. QUESTION-ANSWER EVALUATION
    print("\n3. QUESTION-ANSWER METRICS")
    print("-" * 27)

    qa_column = "answer"
    gt_answers = gt_df[qa_column].astype(str)
    model_answers = model_df[qa_column].astype(str)

    # Calculate BLEU scores
    bleu_scores = []
    for gt_answer, model_answer in zip(gt_answers, model_answers):
        # Handle NA values - treat as empty string for BLEU calculation
        if gt_answer == "nan" or model_answer == "nan":
            if gt_answer == model_answer:  # Both are NA
                bleu_scores.append(1.0)  # Perfect match
            else:
                bleu_scores.append(0.0)  # One is NA, other is not
        elif gt_answer.strip() == "" or model_answer.strip() == "":
            bleu_scores.append(0.0)
        else:
            # Split into words
            gt_words = gt_answer.split()
            model_words = model_answer.split()

            if len(model_words) == 0:
                bleu_scores.append(0.0)
            else:
                bleu = sentence_bleu([gt_words], model_words)
                bleu_scores.append(bleu)

    avg_bleu = np.mean(bleu_scores)
    rouge = Rouge()
    rouge_1_scores = []
    rouge_l_scores = []

    for gt_answer, model_answer in zip(gt_answers, model_answers):
        if gt_answer == "nan" or model_answer == "nan":
            if gt_answer == model_answer:  # Both are NA
                rouge_1_scores.append(1.0)  # Perfect match
                rouge_l_scores.append(1.0)
            else:
                rouge_1_scores.append(0.0)  # One is NA, other is not
                rouge_l_scores.append(0.0)
        elif gt_answer.strip() == "" or model_answer.strip() == "":
            rouge_1_scores.append(0.0)
            rouge_l_scores.append(0.0)
        else:
            try:
                scores = rouge.get_scores(model_answer, gt_answer)
                rouge_1_scores.append(scores[0]["rouge-1"]["f"])
                rouge_l_scores.append(scores[0]["rouge-l"]["f"])
            except:
                rouge_1_scores.append(0.0)
                rouge_l_scores.append(0.0)

    avg_rouge_1 = np.mean(rouge_1_scores)
    avg_rouge_l = np.mean(rouge_l_scores)

    try:
        P, R, F1 = bert_score.score(
            model_answers.tolist(), gt_answers.tolist(), lang="en", verbose=False
        )
        avg_bert_f1 = F1.mean().item()
    except:
        print("Warning: BERTScore calculation failed")
        avg_bert_f1 = 0.0

    print(f"\nFinal Results:")
    print(f"BLEU Score: {avg_bleu:.3f}")
    print(f"ROUGE-1: {avg_rouge_1:.3f}")
    print(f"ROUGE-L: {avg_rouge_l:.3f}")
    print(f"BERTScore F1: {avg_bert_f1:.3f}")


if __name__ == "__main__":
    # Replace with your actual file paths
    ground_truth_csv = "Ground-truth/ground_truth_csv.csv"
    model_output_csv = FILE_DESTINATION_ADDRESS

    compare_csvs(ground_truth_csv, model_output_csv)
