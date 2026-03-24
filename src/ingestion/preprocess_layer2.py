import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict

# Folder containing your raw scraped JSON files
# Each subfolder (Glossary/, Offense/, Defense/) contains .json files
RAW_INPUT = Path("corpus/raw/layer2_hoopstudent.json")

# Where to write the final standardized chunk file
OUTPUT_DIR = Path("corpus/processed")
OUTPUT_FILE = OUTPUT_DIR / "layer2_chunks.json"

# Minimum character length for a section to become a chunk
# Sections shorter than this are merged into the definition chunk or skipped
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


def build_definition_chunk(doc: dict, chunk_index: int) -> HoopStudentChunk:
    """
    Creates a single high-priority chunk from the concise_definition field.

    This chunk is short and dense — it retrieves well for exact queries
    like "what is a pick and roll?" because the term name and definition
    are both in the same small text window.

    Format:
      "{term}: {concise_definition}"
    """
    term = doc["term"]
    definition = doc["concise_definition"].strip()

    # Remove redundant "term : " prefix that the scraper sometimes includes
    # e.g. "1-1-2-1 press defense : Basketball strategy that..."
    # becomes just "Basketball strategy that..."
    clean_def = re.sub(
        r'^' + re.escape(term) + r'\s*[:\-–]\s*',
        '',
        definition,
        flags=re.IGNORECASE
    ).strip()

    # Enriched text: term name is explicit so BM25 matches it
    enriched_text = f"{term}: {clean_def}"

    chunk_id = f"{doc['doc_id']}_definition"

    metadata = {
        "source": "hoopstudent",
        "layer": 2,
        "layer_name": "plays_and_actions",
        "doc_id": doc["doc_id"],
        "term": term,
        "category": doc.get("category", ""),
        "source_site": doc.get("source_site", "hoopstudent"),
        "source_url": doc.get("source_url", ""),
        "chunk_type": "definition",
        "section_heading": "Definition"
    }

    return HoopStudentChunk(
        chunk_id=chunk_id,
        doc_id=doc["doc_id"],
        text=enriched_text,
        metadata=metadata,
    )


def build_section_chunk(
    doc: dict,
    section: dict,
    section_index: int,
) -> HoopStudentChunk | None:
    """
    Creates one chunk from a single section of a HoopStudent document.

    The context prefix prepends the term name and section heading so that
    BM25 can match on the term name even when querying a section-level chunk.

    Format:
      "{term} — {section_heading}:\n\n{section_text}"

    Returns None if the section should be skipped.
    """
    term = doc["term"]
    heading = section.get("heading", "").strip()
    text = section.get("text", "").strip()

    # Skip navigation/boilerplate sections
    if is_skip_heading(heading):
        return None

    # Skip sections with no meaningful content
    if is_noise_section(text):
        return None

    # Build enriched text: term + heading + body
    #  With the prefix, BM25 and the dense
    # embedder both understand the full context.
    if heading:
        enriched_text = f"{term} — {heading}:\n\n{text}"
    else:
        enriched_text = f"{term}:\n\n{text}"

    chunk_id = f"{doc['doc_id']}_section_{section_index:03d}"

    metadata = {
        "source": "hoopstudent",
        "layer": 2,
        "layer_name": "plays_and_actions",
        "doc_id": doc["doc_id"],
        "term": term,
        "category": doc.get("category", ""),
        "source_site": doc.get("source_site", "hoopstudent"),
        "source_url": doc.get("source_url", ""),
        "chunk_type": "section",
        "section_heading": heading
    }

    return HoopStudentChunk(
        chunk_id=chunk_id,
        doc_id=doc["doc_id"],
        text=enriched_text,
        metadata=metadata,
    )

def process_document(doc: dict) -> list[HoopStudentChunk]:
    """
    Takes one raw hoopstudent document and returns a list of chunks.

    One document = one term page on hoopstudent.com
    Output = 1 definition chunk + N section chunks
    """
    chunks = []

    # Always create a definition chunk — even if very short
    # This is the "anchor" chunk for exact-match queries
    if doc.get("concise_definition"):
        def_chunk = build_definition_chunk(doc, chunk_index=0)
        chunks.append(def_chunk)

    # Create one chunk per section
    sections = doc.get("sections", [])
    for i, section in enumerate(sections):
        section_chunk = build_section_chunk(doc, section, section_index=i)
        if section_chunk is not None:
            chunks.append(section_chunk)

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
    print("LAYER 2 PREPROCESSING — HoopStudent")
    print("=" * 60)
    print(f"Input file      : {RAW_INPUT}")
    print(f"Output file     : {OUTPUT_FILE}")

    # Step 1: Load the single raw JSON file
    print(f"\nLoading raw data...")
    documents = load_raw_documents(RAW_INPUT)

    if not documents:
        print("\n[!] No documents found.")
        return

    print(f"Loaded {len(documents)} documents.")

    # Step 2: Process each document into chunks
    print("\nChunking documents...")
    all_chunks: list[HoopStudentChunk] = []

    for doc in documents:
        doc_chunks = process_document(doc)
        all_chunks.extend(doc_chunks)

    # Step 3: Print stats and sample output
    print_stats(all_chunks, documents)

    # Step 4: Save to JSON — same format as layer 1
    output_data = [asdict(chunk) for chunk in all_chunks]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n Saved {len(all_chunks)} chunks to: {OUTPUT_FILE}")
    print(f'  "{OUTPUT_FILE}"')

if __name__ == "__main__":
    main()