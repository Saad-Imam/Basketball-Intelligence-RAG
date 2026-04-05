#SemanticChunker (Greg Kamradt / LlamaIndex approach) 
#       1. Split the section body into sentences (NLTK).
#       2. Embed each sentence using a lightweight bi-encoder
#          (all-MiniLM-L6-v2, 80 MB, already used in evaluate.py).
#       3. Compute cosine distance between consecutive sentence embeddings.
#       4. Identify breakpoints at the Nth percentile of those distances.
#       5. Group sentences between breakpoints → variable-length chunks.
#       6. Cap chunks that exceed MAX_TOKENS by recursive sentence splitting.
import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker as DoclingHybridChunker

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

RULEBOOKS = [
    {
        "path": "corpus/raw/Official-2025-26-NBA-Playing-Rules.pdf",
        "doc_id": "nba_rulebook_2025",
        "source": "nba_rulebook",
        "source_url": "https://ak-static.cms.nba.com/wp-content/uploads/sites/4/2025/10/Official-2025-26-NBA-Playing-Rules.pdf",
        "league": "NBA",
        "skip_pages": set(range(1, 9)) | set(range(72, 77)),
    },
    {
        "path": "corpus/raw/fiba-official-rules-2024-v10a.pdf",
        "doc_id": "fiba_rulebook_2024",
        "source": "fiba_rulebook",
        "source_url": "https://www.fiba.basketball/documents/official-basketball-rules",
        "league": "FIBA",
        "skip_pages": set(range(1, 5)) | set(range(7, 8)) | set(range(62, 71)) | set(range(97, 106)),
    },
]

# Sentence-transformer used ONLY for boundary detection (not for final indexing).
BOUNDARY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Percentile of cosine distances at which we place a chunk boundary.
BREAKPOINT_PERCENTILE = 85

# Hard ceiling — chunks longer than this (in approximate tokens, ~4 chars/token)
# are split further at sentence boundaries.
MAX_TOKENS = 500
MAX_CHARS  = MAX_TOKENS * 4   # quick proxy; accurate enough for rulebook prose

# Sections shorter than this are either merged or discarded as noise.
MIN_SECTION_CHARS = 80

# Context window: Window = 1 means pairs.
SENTENCE_WINDOW = 1

# BGE-M3 tokenizer name — used by DoclingHybridChunker for the section pass.
TOKENIZER_MODEL = "BAAI/bge-m3"

# A very large token ceiling for the section-grouping pass.
# HybridChunker will still split at heading boundaries but won't subdivide
# within a section unless it truly exceeds this limit 
SECTION_MAX_TOKENS = 10_000

OUTPUT_DIR  = Path("corpus/processed")
OUTPUT_FILE = OUTPUT_DIR / "layer1_rulebook_chunks_semantic.json"


@dataclass
class RulebookChunk:
    chunk_id: str
    doc_id:   str
    text:     str          # enriched text (context prefix + body) — this gets embedded
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
  
    def __init__(
        self,
        model: SentenceTransformer,
        breakpoint_percentile: int   = BREAKPOINT_PERCENTILE,
        max_chars:             int   = MAX_CHARS,
        min_chars:             int   = MIN_SECTION_CHARS,
        sentence_window:       int   = SENTENCE_WINDOW,
    ):
        self.model                 = model
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chars             = max_chars
        self.min_chars             = min_chars
        self.sentence_window       = sentence_window

    # ------------------------------------------------------------------
    def chunk(self, text: str) -> list[str]:
        """Main entry point — returns a list of chunk strings."""
        sentences = sent_tokenize(text.strip())

        # Edge cases: very short text → return as-is (or discard)
        if len(sentences) <= 2:
            return [text.strip()] if len(text.strip()) >= self.min_chars else []

        # Step 2: build context windows
        windows = self._build_windows(sentences)

        # Step 3: embed
        embeddings = self.model.encode(windows, normalize_embeddings=True, show_progress_bar=False)

        # Step 4: cosine distances  (normalised vectors → dist = 1 − dot)
        distances = [
            1.0 - float(embeddings[i] @ embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Step 5: breakpoints
        if not distances:
            return self._finalize([text.strip()])

        threshold    = float(np.percentile(distances, self.breakpoint_percentile))
        breakpoints  = {i for i, d in enumerate(distances) if d > threshold}

        # Step 6: group sentences
        raw_chunks = self._group_sentences(sentences, breakpoints)

        # Steps 7 & 8: cap length and drop noise
        return self._finalize(raw_chunks)

    # ------------------------------------------------------------------
    def _build_windows(self, sentences: list[str]) -> list[str]:
        """
        Each window = sentence[i] concatenated with up to `sentence_window`
        following sentences.  This gives the embedder local context.
        """
        windows = []
        n = len(sentences)
        for i in range(n):
            end = min(i + self.sentence_window + 1, n)
            windows.append(" ".join(sentences[i:end]))
        return windows

    def _group_sentences(self, sentences: list[str], breakpoints: set[int]) -> list[str]:
        """Group sentences into chunks at every breakpoint index."""
        chunks = []
        start  = 0
        for bp in sorted(breakpoints):
            group = " ".join(sentences[start : bp + 1]).strip()
            if group:
                chunks.append(group)
            start = bp + 1
        # Remaining sentences after the last breakpoint
        tail = " ".join(sentences[start:]).strip()
        if tail:
            chunks.append(tail)
        return chunks

    def _finalize(self, chunks: list[str]) -> list[str]:
        """
        Apply size constraints:
        - Drop chunks shorter than min_chars.
        - Split chunks longer than max_chars at sentence boundaries.
        """
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
        """
        Fallback: split an oversized chunk greedily at sentence boundaries
        to keep every piece under max_chars.
        """
        sentences = sent_tokenize(text)
        result, current = [], []
        current_len = 0
        for sent in sentences:
            if current_len + len(sent) > self.max_chars and current:
                result.append(" ".join(current).strip())
                current, current_len = [], 0
            current.append(sent)
            current_len += len(sent) + 1
        if current:
            result.append(" ".join(current).strip())
        return [r for r in result if len(r) >= self.min_chars]




def parse_heading_path(headings: list[str]) -> dict:
    """Extract rule/section info from a list of Docling heading strings."""
    result = {
        "rule_number":  None,
        "rule_title":   None,
        "section":      None,
        "section_title": None,
        "chunk_type":   "general",
    }
    for heading in headings:
        h = heading.strip()

        rule_match = re.match(
            r'RULE\s+(?:NO\.?\s+)?(\d+[A-Z]?)\s*[:—\-–\s]\s*(.+)',
            h, re.IGNORECASE
        )
        if rule_match:
            result["rule_number"] = f"Rule {rule_match.group(1)}"
            result["rule_title"]  = rule_match.group(2).strip().title()
            result["chunk_type"]  = "rule_section"
            continue

        section_match = re.match(
            r'Section\s+([IVXLCDM]+)\s*[:—\-–\s]\s*(.+)',
            h, re.IGNORECASE
        )
        if section_match:
            result["section"]       = f"Section {section_match.group(1).upper()}"
            result["section_title"] = section_match.group(2).strip().title()
            continue

        if re.search(r'comments\s+on\s+the\s+rules', h, re.IGNORECASE):
            result["chunk_type"] = "comments"
            continue

        if re.search(r'definitions', h, re.IGNORECASE):
            result["chunk_type"] = "definitions"
            continue

    return result


def build_context_prefix(heading_info: dict, league: str) -> str:
    """
    Prepend a human-readable source label to every chunk.
    Keeps BM25 matching strong on rule-number queries (e.g. "NBA Rule 10").
    """
    parts = [f"{league} Rulebook"]
    if heading_info["rule_number"] and heading_info["rule_title"]:
        parts.append(f"{heading_info['rule_number']} ({heading_info['rule_title']})")
    if heading_info["section"] and heading_info["section_title"]:
        parts.append(f"{heading_info['section']} ({heading_info['section_title']})")
    return " — ".join(parts) + ":\n\n"


def is_noise_chunk(text: str) -> bool:
    """Filter out page-number lines, pure index entries, etc."""
    stripped = text.strip()
    if len(stripped) < MIN_SECTION_CHARS:
        return True
    if re.match(r'^[-\s\d]+$', stripped):
        return True
    if re.match(r'^(Section|Rule|PAGE)\s+', stripped) and len(stripped) < 60:
        return True
    if len(re.findall(r'\.{5,}\s*\d+', stripped)) > 3:
        return True
    return False



def process_rulebook(config: dict, chunker: SemanticChunker) -> list[RulebookChunk]:
    """
    Converts one PDF rulebook into a list of RulebookChunk objects.

    Pipeline:
      1. Docling converts the PDF to a structured document.
      2. DoclingHybridChunker (max_tokens=10_000) groups the document into
         heading-delimited section blocks — still carrying .meta.headings and
         page provenance, with no lossy intermediate format.
      3. SemanticChunker splits each section block's text at topic-shift
         breakpoints detected from sentence-embedding distances.
      4. Each semantic sub-chunk is wrapped as a RulebookChunk with the full
         heading metadata inherited from its parent section block.
    """
    pdf_path   = config["path"]
    doc_id     = config["doc_id"]
    league     = config["league"]
    skip_pages = config.get("skip_pages", set())

    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"  [!] File not found — skipping.")
        return []

    # Step 1: Docling PDF parse 
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        generate_page_images=False,
    )
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )
    print("  Parsing PDF with Docling …")
    result = converter.convert(pdf_path)
    doc    = result.document

    # --- Step 2: HybridChunker as a heading-boundary grouper ---
    # max_tokens is set very high so it splits at heading changes only,
    # not at token limits.  We get section blocks with .meta.headings intact.
    section_chunker = DoclingHybridChunker(
        tokenizer=TOKENIZER_MODEL,
        max_tokens=SECTION_MAX_TOKENS,
        merge_peers=True,
    )
    section_blocks = list(section_chunker.chunk(doc))
    print(f"  Docling produced {len(section_blocks)} section blocks")

    # --- Steps 3 & 4: Semantic chunk each section block ---
    processed_chunks: list[RulebookChunk] = []
    skipped_noise    = 0
    chunk_global_idx = 0

    for sec_idx, block in enumerate(section_blocks):
        block_text = block.text.strip()

        # --- Extract page numbers (identical logic to the hybrid version) ---
        page_numbers = []
        if hasattr(block, "meta") and hasattr(block.meta, "doc_items"):
            for item in block.meta.doc_items:
                if hasattr(item, "prov"):
                    for prov in item.prov:
                        if hasattr(prov, "page_no"):
                            page_numbers.append(prov.page_no)
        page_numbers = sorted(set(page_numbers))

        # Skip pages in the configured skip list
        if page_numbers and all(p in skip_pages for p in page_numbers):
            skipped_noise += 1
            continue

        # --- Extract heading hierarchy (identical logic to the hybrid version) ---
        headings: list[str] = []
        if hasattr(block, "meta") and hasattr(block.meta, "headings"):
            headings = block.meta.headings or []

        heading_info   = parse_heading_path(headings)
        context_prefix = build_context_prefix(heading_info, league)

        # --- Apply SemanticChunker to the section body ---
        sub_chunks = chunker.chunk(block_text)

        for sub_idx, sub_text in enumerate(sub_chunks):
            if is_noise_chunk(sub_text):
                skipped_noise += 1
                continue

            enriched_text = context_prefix + sub_text
            chunk_id      = f"{doc_id}_sem_chunk_{chunk_global_idx:04d}"
            chunk_global_idx += 1

            metadata = {
                "source":          config["source"],
                "layer":           1,
                "layer_name":      "rulebook",
                "doc_id":          doc_id,
                "league":          league,
                "source_url":      config["source_url"],
                "chunk_type":      heading_info["chunk_type"],
                "rule_number":     heading_info["rule_number"],
                "rule_title":      heading_info["rule_title"],
                "section":         heading_info["section"],
                "section_title":   heading_info["section_title"],
                "headings_path":   headings,
                "page_numbers":    page_numbers,
                # Semantic-specific fields
                "chunk_strategy":  "semantic",
                "section_index":   sec_idx,
                "sub_chunk_index": sub_idx,
            }

            processed_chunks.append(RulebookChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=enriched_text,
                metadata=metadata,
            ))

    print(f"  Skipped (noise / skip-pages): {skipped_noise}")
    print(f"  Final semantic chunks: {len(processed_chunks)}")
    return processed_chunks



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LAYER 1 SEMANTIC CHUNKING")
    print("=" * 60)

    print(f"\nLoading boundary-detection model: {BOUNDARY_MODEL_NAME}")
    model   = SentenceTransformer(BOUNDARY_MODEL_NAME)
    chunker = SemanticChunker(
        model=model,
        breakpoint_percentile=BREAKPOINT_PERCENTILE,
        max_chars=MAX_CHARS,
        min_chars=MIN_SECTION_CHARS,
    )
    print("  Model loaded.\n")

    all_chunks: list[RulebookChunk] = []
    for config in RULEBOOKS:
        chunks = process_rulebook(config, chunker)
        all_chunks.extend(chunks)

    # --- Stats ---
    print(f"\n{'='*60}")
    print(f"TOTAL SEMANTIC CHUNKS: {len(all_chunks)}")
    nba  = sum(1 for c in all_chunks if c.metadata.get("league") == "NBA")
    fiba = sum(1 for c in all_chunks if c.metadata.get("league") == "FIBA")
    print(f"  NBA:  {nba}")
    print(f"  FIBA: {fiba}")

    # --- Serialise ---
    output_data = [asdict(c) for c in all_chunks]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(all_chunks)} chunks → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()