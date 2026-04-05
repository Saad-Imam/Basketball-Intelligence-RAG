import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


NAMESPACE = "semantic"   
CHUNK_FILES = [
        "corpus/processed/layer1_rulebook_chunks_semantic.json",
        "corpus/processed/layer2_chunks_semantic.json",
        ]


INDEX_NAME      = "basketball-rag-hybrid-bge"  
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


BGE_MODEL_NAME  = "BAAI/bge-m3"
USE_FP16        = False        # set True if running on GPU

EMBED_BATCH_SIZE  = 4          # safe on CPU; ~2-5 min per 100 chunks
UPSERT_BATCH_SIZE = 100
DENSE_DIMENSION   = 1024



def load_all_chunks(chunk_files: list[str]) -> list[dict]:
    """Each chunk must have: chunk_id, text, metadata."""
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



def generate_embeddings(
    model: BGEM3FlagModel,
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
) -> tuple[list[list[float]], list[dict]]:
    """
    Runs BGE-M3 on a list of texts.

    Returns:
      dense_vectors  — list of 1024-float lists
      sparse_vectors — list of {"indices": [...], "values": [...]} dicts

    BGE-M3 produces both in a SINGLE forward pass.
    """
    dense_vectors  = []
    sparse_vectors = []
    total_batches  = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(texts), batch_size):
        batch_texts   = texts[batch_idx : batch_idx + batch_size]
        current_batch = batch_idx // batch_size + 1

        print(f"  Encoding batch {current_batch}/{total_batches} "
              f"({len(batch_texts)} texts)...", end="\r")

        output = model.encode(
            batch_texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=batch_size,
            max_length=512,
        )

        batch_dense = output["dense_vecs"].tolist()
        dense_vectors.extend(batch_dense)

        for sparse_dict in output["lexical_weights"]:
            filtered = {
                k: float(v)
                for k, v in sparse_dict.items()
                if float(v) > 0.01
            }
        # FIX: Explicitly cast k to int here
            top_tokens = sorted(
                [(int(k), float(v)) for k, v in filtered.items()], # Cast to (int, float)
                key=lambda x: x[1], 
                reverse=True)[:1000]            
            if top_tokens:
                indices, values = zip(*top_tokens)
                sparse_vectors.append({"indices": list(indices), "values": list(values)})
            else:
                sparse_vectors.append({"indices": [], "values": []})

    print(f"\n  Done encoding {len(dense_vectors)} vectors.")
    return dense_vectors, sparse_vectors



def create_pinecone_index(pc: Pinecone, index_name: str) -> None:
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name in existing_indexes:
        print(f"  Index '{index_name}' already exists — skipping creation.")
        return

    print(f"  Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=DENSE_DIMENSION,
        metric="dotproduct",          # required for hybrid search
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    print("  Waiting for index to be ready...")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(2)
    print(f"  Index '{index_name}' is ready.")


def upsert_to_pinecone(
    pc:             Pinecone,
    index_name:     str,
    namespace:      str,
    chunks:         list[dict],
    dense_vectors:  list[list[float]],
    sparse_vectors: list[dict],
    batch_size:     int = UPSERT_BATCH_SIZE,
) -> None:
    """
    Upserts vectors into  index_name / namespace.
    """
    index          = pc.Index(index_name)
    total_batches  = (len(chunks) + batch_size - 1) // batch_size
    total_upserted = 0

    print(f"\nUpserting {len(chunks)} vectors → namespace='{namespace}' "
          f"in batches of {batch_size}...")

    for batch_idx in range(0, len(chunks), batch_size):
        batch_end    = min(batch_idx + batch_size, len(chunks))
        batch_chunks = chunks[batch_idx:batch_end]
        batch_dense  = dense_vectors[batch_idx:batch_end]
        batch_sparse = sparse_vectors[batch_idx:batch_end]

        records = []
        for chunk, dense, sparse in zip(batch_chunks, batch_dense, batch_sparse):
            meta = dict(chunk.get("metadata", {}))
            if "text" not in meta:
                meta["text"] = chunk.get("text", "")

            clean_meta = {}
            for k, v in meta.items():
                if v is None:
                    clean_meta[k] = ""
                elif isinstance(v, list):
                    clean_meta[k] = [str(i) for i in v]
                else:
                    clean_meta[k] = v

            records.append({
                "id":            chunk["chunk_id"],
                "values":        dense,
                "sparse_values": sparse,
                "metadata":      clean_meta,
            })

        # Pass namespace to the upsert call
        index.upsert(vectors=records, namespace=namespace)
        total_upserted += len(records)

        current_batch = batch_idx // batch_size + 1
        print(f"  Batch {current_batch}/{total_batches} — "
              f"upserted {total_upserted}/{len(chunks)} vectors")
        time.sleep(0.1)

    print(f"\n Upsert complete.  Namespace '{namespace}' now has "
          f"~{total_upserted} vectors in this run.")


def verify_index(pc: Pinecone, index_name: str, namespace: str) -> None:
    index = pc.Index(index_name)
    stats = index.describe_index_stats()

    print(f"\n{'='*60}")
    print(f"INDEX VERIFICATION — '{index_name}'")
    print(f"{'='*60}")
    print(f"Total vectors (all namespaces): {stats.total_vector_count}")
    print(f"Dimension:                      {stats.dimension}")

    ns_stats = stats.namespaces or {}
    for ns_name, ns_info in ns_stats.items():
        print(f"  namespace '{ns_name}': {ns_info.vector_count} vectors")


def main():
    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY not found. "
            "Create a .env file with PINECONE_API_KEY=your_key"
        )

    print("=" * 60)
    print(f"EMBEDDING & INDEXING  —  namespace = '{NAMESPACE}'")
    print("=" * 60)

    # Step 1
    print("\nSTEP 1: Loading chunks")
    print("=" * 60)
    all_chunks = load_all_chunks(CHUNK_FILES)
    if not all_chunks:
        print("No chunks found.")
        return
    texts = [chunk["text"] for chunk in all_chunks]

    # Step 2: Load BGE-M3
    print("\nSTEP 2: Loading BGE-M3 model")
    print("=" * 60)
    print(f"  Model: {BGE_MODEL_NAME}  FP16={USE_FP16}")
    model = BGEM3FlagModel(BGE_MODEL_NAME, use_fp16=USE_FP16)
    print("  Model loaded.")

    # Step 3: Embed
    print("\nSTEP 3: Generating embeddings (dense + sparse)")
    print("=" * 60)
    print(f"  Total chunks: {len(texts)}  |  Batch size: {EMBED_BATCH_SIZE}")
    start_time = time.time()
    dense_vectors, sparse_vectors = generate_embeddings(model, texts)
    elapsed = time.time() - start_time
    print(f"  Embedding time: {elapsed:.1f}s  "
          f"({elapsed / len(texts) * 1000:.1f} ms/chunk)")

    # Sanity checks
    assert len(dense_vectors)  == len(all_chunks)
    assert len(sparse_vectors) == len(all_chunks)
    assert len(dense_vectors[0]) == DENSE_DIMENSION
    print(f"  Dense dim:          {len(dense_vectors[0])} ✓")
    print(f"  Sample sparse nnz:  {len(sparse_vectors[0]['indices'])} tokens")

    # Save checkpoint
    checkpoint_path = Path(f"corpus/embeddings_checkpoint_{NAMESPACE}.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving checkpoint → {checkpoint_path} …")
    with open(checkpoint_path, "w") as f:
        json.dump({
            "namespace":     NAMESPACE,
            "chunk_ids":     [c["chunk_id"] for c in all_chunks],
            "dense_vectors": dense_vectors,
            "sparse_vectors": sparse_vectors,
        }, f)
    print("  Checkpoint saved.")

    # Step 4: Pinecone index
    print("\nSTEP 4: Setting up Pinecone index")
    print("=" * 60)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    create_pinecone_index(pc, INDEX_NAME)

    # Step 5: Upsert into the correct namespace
    print("\nSTEP 5: Upserting to Pinecone")
    print("=" * 60)
    upsert_to_pinecone(
        pc=pc,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
        chunks=all_chunks,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
    )
    verify_index(pc, INDEX_NAME, NAMESPACE)

    print(f"\n✓ Indexing complete.  index='{INDEX_NAME}'  namespace='{NAMESPACE}'")


# ---------------------------------------------------------------------------
# Resume from checkpoint (if embedding succeeded but upsert failed)
# ---------------------------------------------------------------------------
def upsert_from_checkpoint():
    """
    Call this instead of main() if:
      - Embeddings were already generated and saved to the checkpoint file.
      - But the Pinecone upsert was interrupted.
    """
    print(f"Loading from checkpoint for namespace='{NAMESPACE}' …")
    all_chunks = load_all_chunks(CHUNK_FILES)

    checkpoint_path = f"corpus/embeddings_checkpoint_{NAMESPACE}.json"
    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)

    dense_vectors  = checkpoint["dense_vectors"]
    sparse_vectors = checkpoint["sparse_vectors"]
    print(f"Loaded {len(dense_vectors)} embeddings from checkpoint.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    create_pinecone_index(pc, INDEX_NAME)
    upsert_to_pinecone(pc, INDEX_NAME, NAMESPACE, all_chunks, dense_vectors, sparse_vectors)
    verify_index(pc, INDEX_NAME, NAMESPACE)


if __name__ == "__main__":
    main()
    # If resuming from a failed upsert, comment out main() and uncomment:
    # upsert_from_checkpoint()