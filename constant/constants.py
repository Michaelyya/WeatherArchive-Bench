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
    "2": "Qwen/Qwen2.5-7B-Instruct",
    "3": "Qwen/Qwen2.5-14B-Instruct",
    "4": "Qwen/Qwen2.5-32B-Instruct",
    "5": "Qwen/Qwen2.5-72B-Instruct",
    "6": "mistralai/Ministral-8B-Instruct-2410",
    "7": "mistralai/Mistral-Small-3.1-24B-Base-2503",
    "8": "google/gemma-2-9b-it",
}
OPENAI_MODELS = {"9": "gpt-3.5-turbo", "10": "gpt-4", "11": "gpt-4o"}
