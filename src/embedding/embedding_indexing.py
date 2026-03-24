"""
pinecone_indexing.py
====================
Loads preprocessed chunk JSON files, generates BGE-M3 dense + sparse
embeddings locally, and upserts everything into a single Pinecone
hybrid index.

What this script does:
  1. Loads all chunk JSON files from the corpus/ directory
     (layer1 rulebook chunks + layer2 hoopstudent chunks, etc.)
  2. Generates BOTH dense and sparse vectors using BGE-M3 in one pass
     (no separate BM25 encoder needed — BGE-M3 does both natively)
  3. Creates a single Pinecone hybrid index (dimension=1024, metric=dotproduct)
  4. Upserts all chunks in batches with their metadata

Usage:
  pip install FlagEmbedding pinecone-client python-dotenv
  python pinecone_indexing.py

Environment variables (create a .env file):
  PINECONE_API_KEY=your_key_here
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# All chunk JSON files to index — add new ones here as you process more sources
CHUNK_FILES = [
    "corpus/processed_rulebook/layer1_rulebook_chunks.json",
    "corpus/processed_hoopstudent/layer2_hoopstudent_chunks.json",
    # "corpus/processed_articles/layer3_strategy_chunks.json",  # add later
]

# Pinecone settings
INDEX_NAME = "basketball-rag"        # name of your index in Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# BGE-M3 settings
# use_fp16=True cuts memory usage roughly in half on GPU with minimal quality loss
# On CPU, set use_fp16=False
BGE_MODEL_NAME = "BAAI/bge-m3"
USE_FP16 = True       # set False if running on CPU only

# Batching
# BGE-M3 is large (~2.2GB). On a GPU (Colab/Kaggle T4):
#   - batch_size=16 works comfortably
#   - batch_size=32 may OOM on 15GB VRAM depending on chunk length
# On CPU:
#   - batch_size=4 is safe, expect ~2-5 minutes per 100 chunks
EMBED_BATCH_SIZE = 16

# Pinecone upsert batch size — keep at 100 (Pinecone's recommended max per request)
UPSERT_BATCH_SIZE = 100

# BGE-M3's dense embedding dimension — fixed, do not change
DENSE_DIMENSION = 1024

# ---------------------------------------------------------------------------
# STEP 1: Load all chunks from JSON files
# ---------------------------------------------------------------------------

def load_all_chunks(chunk_files: list[str]) -> list[dict]:
    """
    Loads and merges chunks from all JSON files.
    Each chunk must have: chunk_id, text, metadata
    """
    all_chunks = []

    for filepath in chunk_files:
        if not os.path.exists(filepath):
            print(f"  [!] File not found, skipping: {filepath}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"  Loaded {len(chunks)} chunks from: {filepath}")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks loaded: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# STEP 2: Generate BGE-M3 embeddings (dense + sparse in one pass)
# ---------------------------------------------------------------------------

def generate_embeddings(
    model: BGEM3FlagModel,
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE
) -> tuple[list[list[float]], list[dict]]:
    """
    Runs BGE-M3 on a list of texts and returns:
      - dense_vectors:  list of 1024-float lists
      - sparse_vectors: list of dicts, each with 'indices' and 'values' lists

    BGE-M3 generates both in a SINGLE forward pass — no extra BM25 encoder needed.
    The sparse output is a learned sparse representation (similar to SPLADE)
    that outperforms traditional BM25 on most benchmarks.

    The return_dense and return_sparse flags tell the model what to compute.
    We do NOT use return_colbert_vecs (multi-vector) — it's overkill for this project.
    """
    dense_vectors = []
    sparse_vectors = []

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx : batch_idx + batch_size]
        current_batch = batch_idx // batch_size + 1

        print(f"  Encoding batch {current_batch}/{total_batches} "
              f"({len(batch_texts)} texts)...", end="\r")

        output = model.encode(
            batch_texts,
            return_dense=True,         # get 1024-dim dense vectors
            return_sparse=True,        # get learned sparse vectors (like SPLADE)
            return_colbert_vecs=False, # skip multi-vector — not needed
            batch_size=batch_size,
            max_length=512,            # match your MAX_TOKENS from preprocessing
        )

        # Dense vectors: shape (batch_size, 1024)
        # Convert from numpy float32 to Python lists for Pinecone
        batch_dense = output["dense_vecs"].tolist()
        dense_vectors.extend(batch_dense)

        # Sparse vectors: BGE-M3 returns a list of dicts like:
        # {"token1_id": weight1, "token2_id": weight2, ...}
        # We need to convert to Pinecone's format: {"indices": [...], "values": [...]}
        batch_sparse_raw = output["lexical_weights"]

        for sparse_dict in batch_sparse_raw:
            # sparse_dict is a DefaultDict mapping token_id (int) -> weight (float)
            # Filter out near-zero weights to keep sparse vectors compact
            # Pinecone supports up to 1000 non-zero values per sparse vector
            filtered = {
                k: float(v)
                for k, v in sparse_dict.items()
                if float(v) > 0.01  # threshold removes truly negligible tokens
            }

            # Sort by weight descending and cap at 1000 (Pinecone's limit)
            top_tokens = sorted(
                filtered.items(), key=lambda x: x[1], reverse=True
            )[:1000]

            if top_tokens:
                indices, values = zip(*top_tokens)
                sparse_vectors.append({
                    "indices": list(indices),
                    "values": list(values)
                })
            else:
                # Fallback: empty sparse vector (shouldn't happen with real text)
                sparse_vectors.append({"indices": [], "values": []})

    print(f"\n  Done encoding {len(dense_vectors)} vectors.")
    return dense_vectors, sparse_vectors


# ---------------------------------------------------------------------------
# STEP 3: Create the Pinecone hybrid index
# ---------------------------------------------------------------------------

def create_pinecone_index(pc: Pinecone, index_name: str) -> None:
    """
    Creates a single Pinecone serverless index configured for hybrid search.

    WHY these exact settings:
    - dimension=1024: BGE-M3's fixed dense output size. Must match exactly.
    - metric="dotproduct": THE ONLY metric that supports sparse-dense hybrid
      queries in Pinecone. Using cosine or euclidean will cause an error at
      query time when you pass sparse_vector.
    - vector_type="dense": This is a dense index that ALSO supports sparse values.
      Do NOT set vector_type="sparse" — that's a separate sparse-only index type.

    The free Pinecone tier gives you 1 index and 2GB storage — enough for ~100k chunks.
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name in existing_indexes:
        print(f"  Index '{index_name}' already exists — skipping creation.")
        return

    print(f"  Creating index '{index_name}'...")
    print(f"    dimension = {DENSE_DIMENSION}")
    print(f"    metric    = dotproduct  (required for hybrid search)")
    print(f"    cloud     = aws / us-east-1")

    pc.create_index(
        name=index_name,
        dimension=DENSE_DIMENSION,     # 1024 for BGE-M3
        metric="dotproduct",           # REQUIRED for hybrid search — do not change
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"         # Pinecone free tier is on AWS us-east-1
        )
    )

    # Wait for index to be ready before upserting
    print("  Waiting for index to be ready...")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(2)

    print(f"  Index '{index_name}' is ready.")


# ---------------------------------------------------------------------------
# STEP 4: Upsert all vectors to Pinecone
# ---------------------------------------------------------------------------

def upsert_to_pinecone(
    pc: Pinecone,
    index_name: str,
    chunks: list[dict],
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict],
    batch_size: int = UPSERT_BATCH_SIZE
) -> None:
    """
    Upserts all chunk vectors to Pinecone in batches.

    Each record in Pinecone has:
    - id:            unique string (chunk_id)
    - values:        dense vector (1024 floats)
    - sparse_values: sparse vector (indices + values)
    - metadata:      dict of fields for filtering and display

    IMPORTANT: Pinecone metadata has a 40KB limit per record.
    The chunk text is stored in metadata["text"] so we can display
    it in the Streamlit UI without a separate database lookup.
    """
    index = pc.Index(index_name)
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    total_upserted = 0

    print(f"\nUpserting {len(chunks)} vectors in batches of {batch_size}...")

    for batch_idx in range(0, len(chunks), batch_size):
        batch_end = min(batch_idx + batch_size, len(chunks))
        batch_chunks = chunks[batch_idx:batch_end]
        batch_dense = dense_vectors[batch_idx:batch_end]
        batch_sparse = sparse_vectors[batch_idx:batch_end]

        records = []
        for chunk, dense, sparse in zip(batch_chunks, batch_dense, batch_sparse):

            # Build the metadata dict
            # We include all fields from the chunk's metadata
            # but ensure 'text' is always present (needed for retrieval display)
            meta = dict(chunk.get("metadata", {}))
            if "text" not in meta:
                meta["text"] = chunk.get("text", "")

            # Pinecone metadata values must be: str, int, float, bool, or list of str
            # Convert any None values to empty strings to avoid upsert errors
            clean_meta = {}
            for k, v in meta.items():
                if v is None:
                    clean_meta[k] = ""
                elif isinstance(v, list):
                    # Convert list items to strings if they aren't already
                    clean_meta[k] = [str(i) for i in v]
                else:
                    clean_meta[k] = v

            record = {
                "id": chunk["chunk_id"],
                "values": dense,
                "sparse_values": sparse,
                "metadata": clean_meta
            }
            records.append(record)

        # Upsert the batch
        index.upsert(vectors=records)
        total_upserted += len(records)

        current_batch = batch_idx // batch_size + 1
        print(f"  Batch {current_batch}/{total_batches} — "
              f"upserted {total_upserted}/{len(chunks)} vectors")

        # Small delay to be polite to the API
        time.sleep(0.1)

    print(f"\n✓ Upsert complete. Total vectors in index: {total_upserted}")


# ---------------------------------------------------------------------------
# STEP 5: Verify the index after upserting
# ---------------------------------------------------------------------------

def verify_index(pc: Pinecone, index_name: str) -> None:
    """
    Prints index stats to confirm everything was upserted correctly.
    Also runs a quick test query to verify hybrid search works.
    """
    index = pc.Index(index_name)
    stats = index.describe_index_stats()

    print(f"\n{'='*60}")
    print(f"INDEX VERIFICATION — '{index_name}'")
    print(f"{'='*60}")
    print(f"Total vectors: {stats.total_vector_count}")
    print(f"Dimension:     {stats.dimension}")
    print(f"Namespaces:    {stats.namespaces}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # Validate API key
    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY not found. "
            "Create a .env file with PINECONE_API_KEY=your_key"
        )

    # -----------------------------------------------------------------------
    # Load chunks
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading chunks")
    print("=" * 60)
    all_chunks = load_all_chunks(CHUNK_FILES)

    if not all_chunks:
        print("No chunks found. Run the preprocessing scripts first.")
        return

    # Extract just the text for embedding (the enriched_text with context prefix)
    texts = [chunk["text"] for chunk in all_chunks]

    # -----------------------------------------------------------------------
    # Load BGE-M3 model
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Loading BGE-M3 model")
    print("=" * 60)
    print(f"  Model: {BGE_MODEL_NAME}")
    print(f"  FP16:  {USE_FP16}  (set False if on CPU)")
    print("  This downloads ~2.2GB on first run...")

    model = BGEM3FlagModel(
        BGE_MODEL_NAME,
        use_fp16=USE_FP16
    )
    print("  Model loaded.")

    # -----------------------------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Generating BGE-M3 embeddings (dense + sparse)")
    print("=" * 60)
    print(f"  Total chunks to embed: {len(texts)}")
    print(f"  Batch size: {EMBED_BATCH_SIZE}")
    print("  (BGE-M3 generates dense AND sparse in one pass — no separate BM25)")

    start_time = time.time()
    dense_vectors, sparse_vectors = generate_embeddings(model, texts)
    elapsed = time.time() - start_time
    print(f"  Embedding time: {elapsed:.1f}s  ({elapsed/len(texts)*1000:.1f}ms per chunk)")

    # Quick sanity check
    assert len(dense_vectors) == len(all_chunks), "Dense vector count mismatch"
    assert len(sparse_vectors) == len(all_chunks), "Sparse vector count mismatch"
    assert len(dense_vectors[0]) == DENSE_DIMENSION, \
        f"Dense dimension mismatch: expected {DENSE_DIMENSION}, got {len(dense_vectors[0])}"

    print(f"  Dense dimension:  {len(dense_vectors[0])} ✓")
    print(f"  Sample sparse non-zeros: {len(sparse_vectors[0]['indices'])} tokens")

    # -----------------------------------------------------------------------
    # Save embeddings locally as a checkpoint
    # This is important: if Pinecone upsert fails halfway, you don't want to
    # re-run the 2+ hour embedding step. Save first, upsert from file if needed.
    # -----------------------------------------------------------------------
    checkpoint_path = Path("corpus/embeddings_checkpoint.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Saving embedding checkpoint to {checkpoint_path}...")
    checkpoint = {
        "chunk_ids": [c["chunk_id"] for c in all_chunks],
        "dense_vectors": dense_vectors,
        "sparse_vectors": sparse_vectors,
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)
    print("  Checkpoint saved.")

    # -----------------------------------------------------------------------
    # Create Pinecone index
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Setting up Pinecone index")
    print("=" * 60)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    create_pinecone_index(pc, INDEX_NAME)

    # -----------------------------------------------------------------------
    # Upsert vectors
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Upserting to Pinecone")
    print("=" * 60)

    upsert_to_pinecone(
        pc=pc,
        index_name=INDEX_NAME,
        chunks=all_chunks,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
    )

    # -----------------------------------------------------------------------
    # Verify
    # -----------------------------------------------------------------------
    verify_index(pc, INDEX_NAME)

    print("\n✓ Indexing complete. Ready to build the query pipeline.")
    print(f"  Index name: '{INDEX_NAME}'")
    print(f"  Next step: build retrieval_pipeline.py")


# ---------------------------------------------------------------------------
# RESUME FROM CHECKPOINT
# If embedding was done but upsert failed, run this instead of main()
# to skip the expensive embedding step
# ---------------------------------------------------------------------------

def upsert_from_checkpoint():
    """
    Call this function instead of main() if:
    - Embeddings were already generated and saved to checkpoint
    - But Pinecone upsert failed or was interrupted
    """
    print("Loading from checkpoint...")

    chunk_files = CHUNK_FILES
    all_chunks = load_all_chunks(chunk_files)

    checkpoint_path = "corpus/embeddings_checkpoint.json"
    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)

    dense_vectors = checkpoint["dense_vectors"]
    sparse_vectors = checkpoint["sparse_vectors"]

    print(f"Loaded {len(dense_vectors)} embeddings from checkpoint.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    create_pinecone_index(pc, INDEX_NAME)
    upsert_to_pinecone(pc, INDEX_NAME, all_chunks, dense_vectors, sparse_vectors)
    verify_index(pc, INDEX_NAME)


if __name__ == "__main__":
    main()
    # If resuming from checkpoint, comment out main() and uncomment:
    # upsert_from_checkpoint()