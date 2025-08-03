from sentence_transformers import CrossEncoder
import tqdm

class ClimateReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
       #use Cross encoder
        print(f"[ClimateReranker] Loading model: {model_name}")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query, docs, top_n=None):
       #rerank single query
        pairs = [(query, doc[2]) for doc in docs]  # extract chunk
        scores = self.model.predict(pairs)        # cross-encoder calcs 

        reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        final = [(score, doc_id, doc, meta, dissim) for (score, (old_score, doc_id, doc, meta, dissim)) in reranked]

        if top_n:
            return final[:top_n]
        return final
