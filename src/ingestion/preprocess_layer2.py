import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from transformers import AutoTokenizer

# Folder containing your raw scraped JSON files
# Each subfolder (Glossary/, Offense/, Defense/) contains .json files
RAW_INPUT = Path("corpus/raw/layer2_hoopstudent.json")

OUTPUT_DIR  = Path("corpus/processed")
OUTPUT_FILE = OUTPUT_DIR / "layer2_chunks_fixed.json"

# Tokenizer — must match the embedding model used in embedding_indexing.py
TOKENIZER_MODEL = "BAAI/bge-m3"

# Fixed-size chunking settings (mirror the values in preprocess_layer1.py)
CHUNK_SIZE    = 400   # tokens per chunk
CHUNK_OVERLAP = 50    # sliding-window overlap between consecutive chunks

# Minimum character length for a section to be included in the full text
MIN_SECTION_CHARS = 80

# Sections to skip entirely 
SKIP_HEADINGS = {
    "introduction",
    "how to comprehend the player roles and diagrams on this page",
    "how to understand the player roles and diagrams on this page",
    "table of contents",
}

# NOTE: this needs to be the same as layer 1
@dataclass
class HoopStudentChunk:
    chunk_id: str
    doc_id: str
    text: str        # enriched text with context prefix — this gets embedded
    metadata: dict = field(default_factory=dict)

def is_skip_heading(heading: str) -> bool:
    # Returns True if this section heading should be skipped.
    return heading.strip().lower() in SKIP_HEADINGS


def is_noise_section(text: str) -> bool:
    # Returns True if a section's text is too short to be meaningful
    return len(text.strip()) < MIN_SECTION_CHARS


def split_into_fixed_chunks(text: str, tokenizer: AutoTokenizer,
                            chunk_size: int = CHUNK_SIZE,
                            overlap: int = CHUNK_OVERLAP,) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end        = min(start + chunk_size, len(token_ids))
        chunk_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        if end >= len(token_ids):
            break
        start += chunk_size - overlap

    return chunks


def process_document_fixed_size(
    doc: dict,
    tokenizer: AutoTokenizer,
) -> list[HoopStudentChunk]:
    """
    Converts one HoopStudent document into fixed-size token chunks.

    All text for a term (concise definition + every non-noise section) is
    concatenated into a single string in reading order, then split with a
    sliding-window tokenizer. 

    The term name is prepended as a context prefix on every chunk so that
    BM25 can still match an exact term query (e.g. "pick and roll") even when
    the term name itself isn't repeated inside the chunk body.
    """
    term = doc["term"]

    # Step 1 
    parts: list[str] = []

    # Concise definition first — densest signal for exact-match queries
    if doc.get("concise_definition"):
        raw_def = doc["concise_definition"].strip()
        # Strip any redundant "term : " prefix the scraper sometimes includes
        clean_def = re.sub(
            r"^" + re.escape(term) + r"\s*[:\-–]\s*",
            "",
            raw_def,
            flags=re.IGNORECASE,
        ).strip()
        parts.append(f"{term}: {clean_def}")

    # Sections in document order
    for section in doc.get("sections", []):
        heading = section.get("heading", "").strip()
        text    = section.get("text",    "").strip()

        if is_skip_heading(heading) or is_noise_section(text):
            continue

        # Include the section heading
        if heading:
            parts.append(f"{heading}:\n{text}")
        else:
            parts.append(text)

    if not parts:
        return []

    full_text = "\n\n".join(parts)
    # Collapse runs of 3+ newlines to avoid wasting tokens on whitespace
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    # Step 2 
    raw_chunks = split_into_fixed_chunks(full_text, tokenizer)

    # Step 3 
    chunks: list[HoopStudentChunk] = []

    for i, chunk_text in enumerate(raw_chunks):
        if len(chunk_text.strip()) < MIN_SECTION_CHARS:
            continue

        # Prepend the term name so every chunk is self-contained for retrieval
        enriched_text = f"{term}:\n\n{chunk_text}"
        chunk_id      = f"{doc['doc_id']}_fschunk_{i:03d}"

        metadata = {
            "source":        "hoopstudent",
            "layer":         2,
            "layer_name":    "plays_and_actions",
            "doc_id":        doc["doc_id"],
            "term":          term,
            "category":      doc.get("category", ""),
            "source_site":   doc.get("source_site", "hoopstudent"),
            "source_url":    doc.get("source_url", ""),
            "chunk_type":    "fixed_size",
            # "chunk_index":   i,
            # "chunk_size":    CHUNK_SIZE,
            # "chunk_overlap": CHUNK_OVERLAP,
            # Kept for schema compatibility with the section-based version
            "section_heading": None,
        }

        chunks.append(HoopStudentChunk(
            chunk_id=chunk_id,
            doc_id=doc["doc_id"],
            text=enriched_text,
            metadata=metadata,
        ))

    return chunks


def load_raw_documents(file_path: Path) -> list[dict]:
    # Loads the single JSON file containing the list of all scraped terms.
    
    if not file_path.exists():
        print(f"[!] Raw data file not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
            
        # Ensure it's a list
        if not isinstance(documents, list):
            print(f"[!] Expected a list of documents in {file_path.name}")
            return []

        # Simple validation
        valid_docs = [d for d in documents if "doc_id" in d and "term" in d]
        return valid_docs

    except Exception as e:
        print(f"  [!] Error reading {file_path.name}: {e}")
        return []

# STATS: print a summary after processing
def print_stats(chunks: list[HoopStudentChunk], documents: list[dict]) -> None:
    print(f"\n{'='*60}")
    print(f"LAYER 2 CHUNK STATISTICS")
    print(f"{'='*60}")
    print(f"Documents processed : {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LAYER 2 PREPROCESSING — HoopStudent  [fixed-size chunking]")
    print("=" * 60)
    print(f"Input file  : {RAW_INPUT}")
    print(f"Output file : {OUTPUT_FILE}")

    # Load tokenizer once
    print(f"\nLoading tokenizer ({TOKENIZER_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    print(f"  Tokenizer ready. Chunk size: {CHUNK_SIZE} tokens, overlap: {CHUNK_OVERLAP} tokens.")

    # Step 1: Load the single raw JSON file
    print(f"\nLoading raw data...")
    documents = load_raw_documents(RAW_INPUT)

    if not documents:
        print("\n[!] No documents found.")
        return

    print(f"Loaded {len(documents)} documents.")

    # Step 2: Process each document into fixed-size chunks
    print("\nChunking documents (fixed-size)...")
    all_chunks: list[HoopStudentChunk] = []

    for doc in documents:
        doc_chunks = process_document_fixed_size(doc, tokenizer)
        all_chunks.extend(doc_chunks)

    # Step 3: Print stats and sample output
    print_stats(all_chunks, documents)

    # Step 4: Save to JSON — same format as layer 1
    output_data = [asdict(chunk) for chunk in all_chunks]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(all_chunks)} fixed-size chunks to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()