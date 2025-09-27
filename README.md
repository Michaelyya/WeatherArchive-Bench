# WeatherArchive-Bench: Benchmarking Retrieval-Augmented Reasoning for Historical Weather Archives

This repository contains constructed datasets and evaluation frameworks for WeatherArchive-Bench. It comprises two tasks: WeatherArchive-Retrieval, which measures a system’s ability to locate historically relevant passages from over one million archival news segments, and WeatherArchive-Assessment, which evaluates whether Large Language Models (LLMs) can classify societal vulnerability and resilience indicators from extreme weather narratives.

## 📁 Project Structure

```
WXImpactRAG/
├── 📁 constant/                      # Configuration and constants
│   ├── climate_framework.py          # IPCC vulnerability framework definitions
│   └── constants.py                  # File paths and model configurations
│
├── 📁 embedding_loaders/             # Data preprocessing and embedding
│   ├── concat.py                     # Text concatenation utilities
│   └── raw_csv/                      # Historical weather data corpus
│       ├── blizzard_English_*.csv    # Blizzard-related documents
│       ├── cold_English_*.csv        # Cold weather documents
│       ├── heat_English_*.csv        # Heat-related documents
│       ├── storm_English_*.csv       # Storm documents
│       └── ...                       # Other weather phenomena
│
├── 📁 data/                   # Ground truth datasets
│   ├── ground_truth_climate.csv      # Climate assessment ground truth
│   ├── QACandidate_Pool.csv          # Question-answer candidate pool
│   └── QACorrect_Passages.csv        # Correct passage annotations
│
├── 📁 WeatherArchive_Retrieval/      # Retrieval evaluation framework
│   ├── output/                       # Retrieval results
│   │   ├── overall.csv               # Comprehensive retrieval metrics
│   │   ├── raw_BM25*.csv             # BM25 variant results
│   │   ├── raw_model_result_*.csv    # Dense retrieval results
│   │   └── ...                       # Other retrieval outputs
│   ├── retriever_eval_*.py           # Retrieval evaluation scripts
│   ├── overall.py                    # Overall evaluation metrics
│   ├── utils.py                      # Utility functions
│   └── README.md                     # Retrieval framework documentation
│
└── 📁 WeatherArchive_Assessment/     # Climate impact assessment
    ├── output/                        # Assessment results
    │   ├── gpt-4o-results.csv        # GPT-4o assessment results
    │   ├── gpt-3.5-turbo-results.csv # GPT-3.5-turbo results
    │   ├── Qwen2.5-*.csv             # Qwen model results
    │   └── ...                       # Other model outputs
    └── src/                          # Assessment source code
        ├── climate_eval.py           # Climate impact evaluation
        ├── MCQ_metrics.py            # Multiple choice metrics
        ├── QA_metrics.py             # Question-answering metrics
        └── rag_eval.py               # RAG evaluation framework
```

## 🔬 Experiments and Evaluation

### WeatherArchive-Retrieval

**Objective**: Evaluate the effectiveness of various retrieval methods for historical weather data.

**Methods Evaluated**:

- **Sparse Retrieval**: BM25 variants (BM25Okapi, BM25Plus) with cross-encoder reranking
- **Dense Retrieval**:
  - SentenceTransformer models (SBERT, SPLADE)
  - ANCE and UniCoil models
  - Qwen embedding models (0.6B, 4B, 8B)
  - OpenAI embedding models (text-embedding-3-large/small, ada-002)
  - Arctic Embed 2.0 and Granite Embedding R2
  - Google Gemini embedding models


### WeatherArchive-Assessment

**Objective**: Evaluate LLM performance in societal vulnerability and resilience assessment related to extreme weather events based on a well-crafted framework referenced from prior meteorological research. 

**Assessment Framework**:

- **Vulnerability Components**:
  - Exposure: Sudden-Onset | Slow-Onset | Compound hazards
  - Sensitivity: Critical | Moderate | Low system dependence
  - Adaptability: Robust | Constrained | Fragile response capacity
- **Resilience Dimensions**:
  - Temporal Scale: Short-term absorptive | Medium-term adaptive | Long-term transformative
  - Functional System: Health | Energy | Food | Water | Transportation | Information
  - Spatial Scale: Local | Regional | National

## 📊 Key Results Summary

### Retrieval Performance Highlights

| Model                    | Recall@100 | nDCG@100  | MRR@100   | BLEU@1    |
| ------------------------ | ---------- | --------- | --------- | --------- |
| **Gemini Embedding 001** | **95.8%**  | **58.8%** | **48.7%** | **51.7%** |
| Arctic Embed 2.0         | 91.0%      | 54.2%     | 44.5%     | 44.2%     |
| BM25Okapi + CE           | 83.0%      | 52.5%     | 44.0%     | 56.5%     |
| OpenAI-3-large           | 89.6%      | 57.1%     | 47.1%     | 50.2%     |
| ANCE                     | 86.6%      | 40.8%     | 29.3%     | 27.6%     |


## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running WeatherArchive-Retrieval

```bash
# BM25 variants with cross-encoder reranking
python -m WeatherArchive_Retrieval.retriever_eval_1

# Dense retrieval models
python -m WeatherArchive_Retrieval.retriever_eval_2  # SBERT, SPLADE
python -m WeatherArchive_Retrieval.retriever_eval_3  # ANCE, UniCoil
python -m WeatherArchive_Retrieval.retriever_eval_4  # Qwen models
python -m WeatherArchive_Retrieval.retriever_eval_5  # OpenAI models
python -m WeatherArchive_Retrieval.retriever_eval_6  # Arctic, Granite
python -m WeatherArchive_Retrieval.retriever_eval_7  # Gemini models

# Generate overall evaluation metrics
python -m WeatherArchive_Retrieval.overall
```

### Running WeatherArchive-Assessment
```bash
# Societal Vulnerability and Resilience Indicator Classification
python -m WeatherArchive_Assessment.src.climate_eval
# Data analyze
python -m WeatherArchive_Assessment.src.classification_metrics

# Free-form Question Answering 
python -m WeatherArchive_Assessment.src.rag_eval
# Data analyze
python -m WeatherArchive_Assessment.src.QA_metrics
```

## 📝 Data Requirements

- **Input Data**: Historical weather documents in CSV format with 'Text' column
- **Queries**: Question dataset with 'query' column
- **Ground Truth**: Correct passages for evaluation
- **API Keys**: OpenAI, Google, HuggingFace (for respective models)

## 🔧 Configuration

- Model configurations in `constant/constants.py`
- Climate framework definitions in `constant/climate_framework.py`
- File paths and evaluation parameters are customizable



---

_This repository contains the complete implementation and evaluation framework for WeatherArchive-Bench_
