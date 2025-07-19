FILE_1386_ADDRESS = "./Ground-truth/MixedCTX_Dataset(1386).csv"
FILE_CANDIDATE_POOL_ADDRESS = "./Ground-truth/QACandidate_Pool.csv"
FILE_DESTINATION_ADDRESS = "./Ground-truth/generated_structured_answers.csv"

OPENAI_TOKENIZER = "cl100k_base"
OPENAI_BASEMODEL = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

CHROMADB_COLLECTION_NAME = "weather_records"
CHROMADB_CLIENT_ADDRESS = "weather_chroma_store"

LANGUAGE_ENGLISH = "english"

HF_MODELS = {
    "1": "meta-llama/Meta-Llama-3-8B-Instruct",
    "2": "Qwen/Qwen2.5-7B-Instruct",
    "3": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "4": "Qwen/Qwen2.5-14B-Instruct",
    "5": "google/gemma-2-9b-it",
    "6": "mistralai/Mistral-Small-24B-Instruct-2501",
}
OPENAI_MODELS = {"7": "gpt-3.5-turbo", "8": "gpt-4"}
