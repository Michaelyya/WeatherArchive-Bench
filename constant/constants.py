FILE_1386_ADDRESS = "Ground-truth/MixedCTX_Dataset(1386).csv"
FILE_CANDIDATE_POOL_ADDRESS = "Ground-truth/QACandidate_Pool.csv"
FILE_DESTINATION_ADDRESS = "Ground-truth/generated_structured_answers.csv"

OPENAI_TOKENIZER = "cl100k_base"
OPENAI_BASEMODEL = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

CHROMADB_COLLECTION_NAME = "weather_records"
CHROMADB_CLIENT_ADDRESS = "weather_chroma_store"

LANGUAGE_ENGLISH = "english"

HF_MODELS = {
    "1": "meta-llama/Meta-Llama-3-8B-Instruct",
    "2": "meta-llama/Llama-3.3-70B-Instruct",
    "3": "Qwen/Qwen2.5-7B-Instruct",
    "4": "Qwen/Qwen2.5-14B-Instruct",
    "5": "Qwen/Qwen2.5-32B-Instruct",
    "6": "Qwen/Qwen2.5-72B-Instruct",
    "7": "mistralai/Ministral-8B-Instruct-2410",
    "8": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "9": "google/gemma-3-4b-it",
    "10": "google/gemma-3-27b-it"
}
OPENAI_MODELS = {"11": "gpt-3.5-turbo", "12": "gpt-4", "13": "gpt-4o"}
