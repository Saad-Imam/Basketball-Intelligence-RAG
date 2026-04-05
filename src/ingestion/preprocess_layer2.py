import json
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


RAW_INPUT   = Path("corpus/raw/layer2_hoopstudent.json")
OUTPUT_DIR  = Path("corpus/processed")
OUTPUT_FILE = OUTPUT_DIR / "layer2_chunks_semantic.json"

BOUNDARY_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
BREAKPOINT_PERCENTILE = 85    # percentile of cosine distances → chunk boundary
MAX_TOKENS            = 500
MAX_CHARS             = MAX_TOKENS * 4
MIN_SECTION_CHARS     = 80    # drop chunks shorter than this

# Section headings that carry no retrieval value
SKIP_HEADINGS = {
    "introduction",
    "how to comprehend the player roles and diagrams on this page",
    "how to understand the player roles and diagrams on this page",
    "table of contents",
}


@dataclass
class HoopStudentChunk:
    chunk_id: str
    doc_id:   str
    text:     str        # enriched text with context prefix — this gets embedded
    metadata: dict = field(default_factory=dict)



class SemanticChunker:
    """
    Variable-length semantic chunker using percentile-based breakpoint detection.

    See preprocess_layer1_semantic.py for full algorithm commentary.
    """

    def __init__(
        self,
        model:                 SentenceTransformer,
        breakpoint_percentile: int = BREAKPOINT_PERCENTILE,
        max_chars:             int = MAX_CHARS,
        min_chars:             int = MIN_SECTION_CHARS,
        sentence_window:       int = 1,
    ):
        self.model                 = model
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chars             = max_chars
        self.min_chars             = min_chars
        self.sentence_window       = sentence_window

    def chunk(self, text: str) -> list[str]:
        sentences = sent_tokenize(text.strip())
        if len(sentences) <= 2:
            return [text.strip()] if len(text.strip()) >= self.min_chars else []

        windows    = self._build_windows(sentences)
        embeddings = self.model.encode(windows, normalize_embeddings=True, show_progress_bar=False)
        distances  = [
            1.0 - float(embeddings[i] @ embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        if not distances:
            return self._finalize([text.strip()])

        threshold   = float(np.percentile(distances, self.breakpoint_percentile))
        breakpoints = {i for i, d in enumerate(distances) if d > threshold}
        raw_chunks  = self._group_sentences(sentences, breakpoints)
        return self._finalize(raw_chunks)

    def _build_windows(self, sentences: list[str]) -> list[str]:
        n = len(sentences)
        return [
            " ".join(sentences[i : min(i + self.sentence_window + 1, n)])
            for i in range(n)
        ]

    def _group_sentences(self, sentences: list[str], breakpoints: set[int]) -> list[str]:
        chunks, start = [], 0
        for bp in sorted(breakpoints):
            group = " ".join(sentences[start : bp + 1]).strip()
            if group:
                chunks.append(group)
            start = bp + 1
        tail = " ".join(sentences[start:]).strip()
        if tail:
            chunks.append(tail)
        return chunks

    def _finalize(self, chunks: list[str]) -> list[str]:
        result = []
        for chunk in chunks:
            if len(chunk) < self.min_chars:
                continue
            if len(chunk) <= self.max_chars:
                result.append(chunk)
            else:
                result.extend(self._hard_split(chunk))
        return result

    def _hard_split(self, text: str) -> list[str]:
        sentences = sent_tokenize(text)
        result, current, current_len = [], [], 0
        for sent in sentences:
            if current_len + len(sent) > self.max_chars and current:
                result.append(" ".join(current).strip())
                current, current_len = [], 0
            current.append(sent)
            current_len += len(sent) + 1
        if current:
            result.append(" ".join(current).strip())
        return [r for r in result if len(r) >= self.min_chars]


def is_skip_heading(heading: str) -> bool:
    return heading.strip().lower() in SKIP_HEADINGS


def clean_definition(term: str, raw_definition: str) -> str:
    """
    Strip any "term : " prefix the scraper sometimes prepends to the definition.
    e.g. "1-2-1-1 press : Basketball strategy …" → "Basketball strategy …"
    """
    cleaned = re.sub(
        r'^' + re.escape(term) + r'\s*[:\-–]\s*',
        '',
        raw_definition.strip(),
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def build_definition_chunk(doc: dict) -> HoopStudentChunk:
    """
    Short, high-priority anchor chunk built from concise_definition.
    Kept as-is (no semantic re-chunking) because it's already compact and
    retrieves well for "what is X?" queries.
    """
    term       = doc["term"]
    definition = clean_definition(term, doc["concise_definition"])

    return HoopStudentChunk(
        chunk_id=f"{doc['doc_id']}_definition",
        doc_id=doc["doc_id"],
        text=f"{term}: {definition}",
        metadata={
            "source":          "hoopstudent",
            "layer":           2,
            "layer_name":      "plays_and_actions",
            "doc_id":          doc["doc_id"],
            "term":            term,
            "category":        doc.get("category", ""),
            "source_site":     doc.get("source_site", "hoopstudent"),
            "source_url":      doc.get("source_url", ""),
            "chunk_type":      "definition",
            "section_heading": "Definition",
            "chunk_strategy":  "semantic",   # definition anchor, not split
        },
    )

def assemble_body_text(doc: dict) -> str:
    """
    Concatenate all non-skip sections into one prose block for the
    semantic chunker.  Each section is separated by a blank line so
    NLTK's sentence tokeniser doesn't merge sentences across sections.
    """
    parts = []
    for section in doc.get("sections", []):
        heading = section.get("heading", "").strip()
        text    = section.get("text", "").strip()

        if is_skip_heading(heading) or not text:
            continue
        if len(text) < MIN_SECTION_CHARS:
            continue

        # Prepend the section heading as a sentence so the semantic chunker
        # is aware of topic changes signalled by headings.
        # This also improves BM25 matching on heading keywords.
        if heading:
            parts.append(f"{heading}. {text}")
        else:
            parts.append(text)

    return "\n\n".join(parts)


def process_document(doc: dict, chunker: SemanticChunker) -> list[HoopStudentChunk]:
    """
    Converts one scraped HoopStudent document into semantic chunks.

    Returns:
      - 1 definition anchor chunk (always, if concise_definition exists)
      - N semantic chunks from the concatenated body text
    """
    chunks: list[HoopStudentChunk] = []
    term = doc["term"]

    if doc.get("concise_definition"):
        chunks.append(build_definition_chunk(doc))

    # --- Semantic body chunks ---
    body_text = assemble_body_text(doc)
    if not body_text or len(body_text) < MIN_SECTION_CHARS:
        return chunks   

    sub_chunks = chunker.chunk(body_text)

    for idx, sub_text in enumerate(sub_chunks):
        chunk_id = f"{doc['doc_id']}_sem_{idx:03d}"

        # Context prefix:  "pick and roll — Body:\n\n{text}"
        # The prefix is short to leave room for the chunk content.
        enriched_text = f"{term} — Body:\n\n{sub_text}"

        chunks.append(HoopStudentChunk(
            chunk_id=chunk_id,
            doc_id=doc["doc_id"],
            text=enriched_text,
            metadata={
                "source":         "hoopstudent",
                "layer":          2,
                "layer_name":     "plays_and_actions",
                "doc_id":         doc["doc_id"],
                "term":           term,
                "category":       doc.get("category", ""),
                "source_site":    doc.get("source_site", "hoopstudent"),
                "source_url":     doc.get("source_url", ""),
                "chunk_type":     "semantic_body",
                "chunk_strategy": "semantic",
                "sub_chunk_index": idx,
            },
        ))

    return chunks


def load_raw_documents(file_path: Path) -> list[dict]:
    if not file_path.exists():
        print(f"[!] Raw data file not found: {file_path}")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
        if not isinstance(documents, list):
            print("[!] Expected a list of documents.")
            return []
        return [d for d in documents if "doc_id" in d and "term" in d]
    except Exception as e:
        print(f"[!] Error reading {file_path.name}: {e}")
        return []




def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LAYER 2 SEMANTIC CHUNKING — HoopStudent")
    print("=" * 60)
    print(f"Input:  {RAW_INPUT}")
    print(f"Output: {OUTPUT_FILE}")

    documents = load_raw_documents(RAW_INPUT)
    if not documents:
        print("[!] No documents found. Exiting.")
        return
    print(f"\nLoaded {len(documents)} documents.")

    print(f"\nLoading boundary-detection model: {BOUNDARY_MODEL_NAME} …")
    model   = SentenceTransformer(BOUNDARY_MODEL_NAME)
    chunker = SemanticChunker(
        model=model,
        breakpoint_percentile=BREAKPOINT_PERCENTILE,
        max_chars=MAX_CHARS,
        min_chars=MIN_SECTION_CHARS,
    )
    print("  Model loaded.\n")

    all_chunks: list[HoopStudentChunk] = []
    for i, doc in enumerate(documents, 1):
        doc_chunks = process_document(doc, chunker)
        all_chunks.extend(doc_chunks)
        if i % 20 == 0 or i == len(documents):
            print(f"  Processed {i}/{len(documents)} documents  "
                  f"({len(all_chunks)} chunks so far)")

    # --- Stats ---
    definitions  = sum(1 for c in all_chunks if c.metadata.get("chunk_type") == "definition")
    body_chunks  = sum(1 for c in all_chunks if c.metadata.get("chunk_type") == "semantic_body")

    print(f"\n{'='*60}")
    print(f"LAYER 2 SEMANTIC CHUNK STATISTICS")
    print(f"{'='*60}")
    print(f"Documents processed : {len(documents)}")
    print(f"Total chunks        : {len(all_chunks)}")
    print(f"  Definition anchors: {definitions}")
    print(f"  Semantic body     : {body_chunks}")
    if len(documents) > 0:
        print(f"  Avg body chunks / term: {body_chunks / len(documents):.1f}")

    # --- Serialise ---
    output_data = [asdict(c) for c in all_chunks]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(all_chunks)} chunks → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()