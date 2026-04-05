import os
import numpy as np
from typing import Optional
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from pinecone import Pinecone
from sentence_transformers import CrossEncoder

load_dotenv()

INDEX_NAME        = "basketball-rag-hybrid-bge"
BGE_MODEL_NAME    = "BAAI/bge-m3"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")

# How many candidates to pull from each Pinecone search
TOP_K_DENSE   = 20
TOP_K_SPARSE  = 20

# RRF smoothing constant 
RRF_K = 60

# MMR: how many RRF results to consider, and the relevance/diversity trade-off
MMR_CANDIDATES = 15   # take top-15 from RRF into MMR
MMR_LAMBDA     = 0.7
MMR_SELECT     = 10   # how many diverse docs MMR hands to the CrossEncoder

# How many final results to return after CrossEncoder re-ranking
FINAL_TOP_N = 5

class BasketballRetriever:
    """
    Loads BGE-M3 and a CrossEncoder once at startup, then exposes a single
    .retrieve(query) method that runs the full 5-stage pipeline.
    """

    def __init__(self, use_fp16: bool = False, verbose: bool = True):
        self.verbose = verbose
        self._log("Loading BGE-M3 model...")
        self.bge = BGEM3FlagModel(BGE_MODEL_NAME, use_fp16=use_fp16)

        self._log("Loading CrossEncoder...")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        self._log("Connecting to Pinecone...")
        if not PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY not found"
            )
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(INDEX_NAME)
        self._log("Retriever ready.\n")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Retriever] {msg}")

    def _encode_query(self, query: str) -> tuple[list[float], dict]:
        """
        Encodes the query with BGE-M3 in a single forward pass.

        Returns dense and sparse separately since we are using RRF
        """
        output = self.bge.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=1,
            max_length=512,
        )

        dense = output["dense_vecs"][0].tolist()

        # Convert BGE-M3's lexical weights to Pinecone sparse format
        # (same conversion used in embedding_indexing.py)
        # lexical weights: are importance scores assigned to individual tokens (words/subwords) based on how relevant 
        # they are to your query
        sparse_raw = output["lexical_weights"][0]
        filtered = {
            k: float(v)
            for k, v in sparse_raw.items()
            if float(v) > 0.01
        }
        # refer to docs if confused regarding this line
        top_tokens = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:1000]
        if top_tokens:
            indices, values = zip(*top_tokens) # zip basically transposes the list of tuples into two lists: 
            # one for indices and one for values, which are the formats Pinecone expects for sparse vectors.
            sparse = {"indices": [int(i) for i in indices], "values": list(values)}
        else:
            sparse = {"indices": [], "values": []}

        return dense, sparse

    def _pinecone_query(self,dense: list[float], sparse: dict, alpha: float, top_k: int,
    metadata_filter: Optional[dict] = None, ) -> list:
        """
        Queries Pinecone with alpha controlling the dense/sparse balance
        metadata_filter: Optional Pinecone metadata filter dict,
                             e.g. {"league": {"$eq": "NBA"}}
        """
        # Scale each leg by alpha
        scaled_dense  = [v * alpha for v in dense]
        scaled_sparse = {
            "indices": sparse["indices"],
            "values":  [v * (1.0 - alpha) for v in sparse["values"]],
        }

        query_kwargs = dict(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=top_k,
            include_metadata=True,
        )
        if metadata_filter:
            query_kwargs["filter"] = metadata_filter

        return self.index.query(**query_kwargs).matches

    def _rrf(self,dense_results: list, sparse_results: list, k: int = RRF_K,) -> list[dict]:
        """
        Applies RRF
        Returns:
            List of dicts sorted by rrf_score descending, each with:
                chunk_id, rrf_score, text, metadata
        """
        rrf_scores: dict[str, float] = {}
        meta_store: dict[str, dict]  = {}

        for rank, match in enumerate(dense_results):
            cid = match.id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            meta_store[cid] = match.metadata

        for rank, match in enumerate(sparse_results):
            cid = match.id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            meta_store[cid] = match.metadata           # may overwrite, same data

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        return [
            {
                "chunk_id":  cid,
                "rrf_score": rrf_scores[cid],
                "text":      meta_store[cid].get("text", ""),
                "metadata":  meta_store[cid],
            }
            for cid in sorted_ids
        ]

    # Stage 4 — Maximal Marginal Relevance
    def _mmr(
        self,
        query_dense: list[float],
        rrf_results: list[dict],
        top_n: int = MMR_SELECT,
        lambda_param: float = MMR_LAMBDA,
    ) -> list[dict]:
        """
        Applies MMR for diversity.
        We need actual dense vectors for every candidate to compute
        inter-document cosine similarity.  Because Pinecone doesn't return
        stored vectors in query results, we re-embed the retrieved texts here.
        The batch is small (≤ MMR_CANDIDATES ≈ 15 docs), so this is fast.
        CONFIRM THIS FROM SIR!

        BGE-M3 outputs L2-normalised vectors, so cosine similarity = dot product.
        """
        if not rrf_results:
            return []

        texts = [r["text"] for r in rrf_results]

        # Re-embed candidate texts — dense only, no sparse needed
        self._log(f"MMR: re-embedding {len(texts)} candidates for diversity scoring...")
        enc = self.bge.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            batch_size=8,
            max_length=512,
        )
        chunk_vecs = enc["dense_vecs"]          # shape (n, 1024), L2-normalised
        query_vec  = np.array(query_dense)      # shape (1024,)

        # Relevance scores: cosine sim between each candidate and the query
        relevance = chunk_vecs @ query_vec      # shape (n,)

        selected   = []          # indices of chosen docs
        remaining  = list(range(len(rrf_results)))

        while len(selected) < top_n and remaining:
            if not selected:
                # First pick: simply take the most relevant doc
                best_idx = max(remaining, key=lambda i: relevance[i])
            else:
                selected_vecs = chunk_vecs[selected]   # shape (s, 1024)
                best_score = -np.inf
                best_idx   = None

                for i in remaining:
                    rel = float(relevance[i])
                    # Maximum similarity to any already-selected document
                    redundancy = float((chunk_vecs[i] @ selected_vecs.T).max())
                    mmr_score  = lambda_param * rel - (1.0 - lambda_param) * redundancy

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx   = i

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [rrf_results[i] for i in selected]

    def _cross_encoder_rerank( self, query: str, candidates: list[dict], final_n: int = FINAL_TOP_N,) -> list[dict]:

        # Applies CrossEncoder on each (query, chunk)

        pairs     = [(query, c["text"]) for c in candidates]
        ce_scores = self.cross_encoder.predict(pairs)   # numpy array of floats

        for candidate, score in zip(candidates, ce_scores):
            candidate["cross_encoder_score"] = float(score)

        reranked = sorted(
            candidates,
            key=lambda x: x["cross_encoder_score"],
            reverse=True,
        )
        return reranked[:final_n]

    def retrieve(self, query: str,
        top_k_dense:   int   = TOP_K_DENSE,
        top_k_sparse:  int   = TOP_K_SPARSE,
        mmr_candidates: int  = MMR_CANDIDATES,
        mmr_lambda:    float = MMR_LAMBDA,
        mmr_select:    int   = MMR_SELECT,
        final_n:       int   = FINAL_TOP_N,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        End-to-end retrieval pipeline.
        Returns:
            List of result dicts, sorted by CrossEncoder score descending.
        """
        self._log(f"Query: {query!r}")
        dense, sparse = self._encode_query(query)

        self._log(f"Dense search  (alpha=1.0, top_k={top_k_dense})...")
        dense_results  = self._pinecone_query(dense, sparse, alpha=1.0,top_k=top_k_dense,metadata_filter=metadata_filter)

        self._log(f"Sparse search (alpha=0.0, top_k={top_k_sparse})...")
        sparse_results = self._pinecone_query(dense, sparse, alpha=0.0,top_k=top_k_sparse,metadata_filter=metadata_filter)

        self._log("RRF fusion...")
        rrf_results = self._rrf(dense_results, sparse_results, k=RRF_K)
        self._log(f"  {len(rrf_results)} unique candidates after RRF")
        
        # implementing MMR:
        rrf_pool = rrf_results[:mmr_candidates]

        self._log(f"MMR (lambda={mmr_lambda}, selecting {mmr_select} from {len(rrf_pool)})...")
        # mmr_results = self._mmr(query_dense=dense,rrf_results=rrf_pool,top_n=mmr_select,lambda_param=mmr_lambda,)
        mmr_results = self._mmr(query_dense=dense,rrf_results=rrf_pool,top_n=mmr_select,lambda_param=mmr_lambda,)


        self._log(f"CrossEncoder re-ranking {len(mmr_results)} candidates → top {final_n}...")
        final = self._cross_encoder_rerank(query, mmr_results, final_n=final_n)

        self._log(f"Done. Returning {len(final)} results.")
        return final


# Convenience function — import this in app.py / generator.py
_retriever_singleton: Optional[BasketballRetriever] = None

def get_retriever(use_fp16: bool = False) -> BasketballRetriever:
    """
    Returns a cached retriever instance, ensuring that only one instance of the retreiver is created
    and reused across the application.
    """
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = BasketballRetriever(use_fp16=use_fp16)
    return _retriever_singleton


# ---------------------------------------------------------------------------
# Quick smoke-test  —  run:  python src/retrieval/retriever.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = BasketballRetriever(use_fp16=False, verbose=True)

    TEST_QUERIES = [
        "What is the rule for defensive three seconds in the NBA?",
        "How does a pick and roll work?",
        "What happens when the ball goes out of bounds in FIBA rules?",
    ]

    for query in TEST_QUERIES:
        results = retriever.retrieve(query, final_n=5)

        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            league     = meta.get("league", meta.get("source", ""))
            layer_name = meta.get("layer_name", "")
            # rule       = meta.get("rule_number", "")
            # rule_title = meta.get("rule_title", "")
            section    = meta.get("section", "")
            term       = meta.get("term", "")       # Layer 2 field

            # # Build a readable label depending on which layer the chunk came from
            # if layer_name == "rulebook":
            #     loc = f"{league} | {rule} {rule_title} | {section}".strip(" |")
            # else:
            #     loc = f"HoopStudent | {term} | {meta.get('chunk_type', '')}"

            print(f"\n  [{i}] CE={r['cross_encoder_score']:+.4f} | RRF={r['rrf_score']:.4f}")
            print(f"       {loc}")
            print(f"       {r['text'][:220].strip()}...")