import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker

# Config
# NOTE: run this from repo root directory
RULEBOOKS = [
    {
        "path": "corpus/raw_rulebook/Official-2025-26-NBA-Playing-Rules.pdf", 
        "doc_id": "nba_rulebook_2025",
        "source": "nba_rulebook",
        "source_url": "https://ak-static.cms.nba.com/wp-content/uploads/sites/4/2025/10/Official-2025-26-NBA-Playing-Rules.pdf",
        "league": "NBA",
        # Rules Index: 2-7, Court Diagram: 8, Referee Hand Signals: 72-75
        "skip_pages": set(range(1, 9)) | set(range(72, 77)),
    },
    # Q: do we keep the appendixes in the fiba doc? currently they are still there
    {
        "path": "corpus/raw_rulebook/fiba-official-rules-2024-v10a.pdf",
        "doc_id": "fiba_rulebook_2024",
        "source": "fiba_rulebook",
        "source_url": "https://www.fiba.basketball/documents/official-basketball-rules",
        "league": "FIBA",
        "skip_pages": set(range(1, 5)) | set(range(7, 8)) | set(range(62, 71)) | set(range(97, 106)),
    },
    # {
    #     "path": "pdfs/FIBA-Basketball-Rules-Interpretations.pdf",
    #     "doc_id": "fiba_interpretations_2024",
    #     "source": "fiba_interpretations",
    #     "source_url": "https://www.fiba.basketball/documents/official-basketball-rules",
    #     "league": "FIBA",
    #     "skip_pages": set(range(1, 4)),
    # },
]

# HybridChunker settings
TOKENIZER_MODEL = "BAAI/bge-m3"
MAX_TOKENS = 500          # max tokens per chunk — keeps chunks under embedding limit
MIN_SECTION_CHARS = 80    # skip sections shorter than this (

OUTPUT_DIR = Path("corpus/processed_rulebook")
OUTPUT_FILE = OUTPUT_DIR / "layer1_rulebook_chunks.json"

# A single chunk is defined as:

@dataclass
class RulebookChunk:
    chunk_id: str
    doc_id: str
    text: str                    # chunk text WITH context prefix — this gets embedded
    metadata: dict = field(default_factory=dict)

def parse_heading_path(headings: list[str]) -> dict:
    """
    The rulebooks have defined rules/sections headings which are defined in docling, for extracting in a similar order
    Docling returns headings ordered from outermost to innermost.
    e.g. ["RULE NO. 4—DEFINITIONS", "Section II—Dribble"]
    """
    result = {
        "rule_number": None,
        "rule_title": None,
        "section": None,
        "section_title": None,
        "chunk_type": "general",
    }

    for heading in headings:
        heading_clean = heading.strip()

        # Match "RULE NO. 4—DEFINITIONS" or "RULE NO. 12B—FOULS AND PENALTIES", for retrieval based on rulenames
        rule_match = re.match(
            r'RULE\s+NO\.?\s+(\d+[A-Z]?)\s*[—\-–]\s*(.+)',
            heading_clean, re.IGNORECASE
        )
        if rule_match:
            result["rule_number"] = f"Rule {rule_match.group(1)}"
            result["rule_title"] = rule_match.group(2).strip().title()
            result["chunk_type"] = "rule_section"
            continue

        # Match "Section I—The Game Officials" or "Section XIV—Suspension of Play", for retrieval based on sections
        section_match = re.match(
            r'Section\s+([IVXLCDM]+)\s*[—\-–]\s*(.+)',
            heading_clean, re.IGNORECASE
        )
        if section_match:
            result["section"] = f"Section {section_match.group(1).upper()}"
            result["section_title"] = section_match.group(2).strip().title()
            continue

        # Comments on the Rules section
        if re.search(r'comments\s+on\s+the\s+rules', heading_clean, re.IGNORECASE):
            result["chunk_type"] = "comments"
            continue

        # Definitions section
        if re.search(r'definitions', heading_clean, re.IGNORECASE):
            result["chunk_type"] = "definitions"
            continue

    return result


def build_context_prefix(heading_info: dict, league: str) -> str:
    """
    Builds a natural-language prefix prepended to every chunk text.
    This dramatically improves BM25 matching on rule-specific queries
    like 'Rule 4 traveling' or 'NBA defensive three seconds'.

    Example output:
      "NBA Rulebook — Rule 10 (Violations And Penalties), Section XIII (Traveling):\n\n"
    """
    parts = [f"{league} Rulebook"]

    if heading_info["rule_number"] and heading_info["rule_title"]:
        parts.append(
            f"{heading_info['rule_number']} ({heading_info['rule_title']})"
        )

    if heading_info["section"] and heading_info["section_title"]:
        parts.append(
            f"{heading_info['section']} ({heading_info['section_title']})"
        )

    prefix = " — ".join(parts) + ":\n\n"
    return prefix


def is_noise_chunk(text: str) -> bool:
    """
    Filters out chunks that are pure navigation or have no retrieval value.
    - Very short chunks (just a heading with no body)
    - Chunks that are entirely page numbers or dashes
    - Chunks that are just "PAGE" markers from the index

    Note that some rulebooks like FIBA have diagrams alongside text, so we need to make sure diagrams don't leave stray text behind
    leading to useless chunks. Docling's OCR is pretty good at ignoring diagrams, but we add this as an extra safety net.
    """
    stripped = text.strip()
    if len(stripped) < MIN_SECTION_CHARS:
        return True

    # Pure page number line  e.g. "- 9 -"
    if re.match(r'^[-\s\d]+$', stripped):
        return True

    # Index entry patterns (just "Rule No. X ... Page Y")
    if re.match(r'^(Section|Rule|PAGE)\s+', stripped) and len(stripped) < 60:
        return True
    # If the text has multiple lines with dot leaders ending in numbers 
    if len(re.findall(r'\.{5,}\s*\d+', stripped)) > 3:
        return True

    return False

def process_rulebook(config: dict) -> list[RulebookChunk]:
    """
    Converts one PDF to a list of RulebookChunk objects using Docling.
    """
    pdf_path = config["path"]
    doc_id = config["doc_id"]
    league = config["league"]
    skip_pages = config.get("skip_pages", set())

    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"Skipping pages: {sorted(skip_pages)}")

    if not os.path.exists(pdf_path):
        print(f"  [!] File not found: {pdf_path} — skipping.")
        return []

    # Step 1: Configure Docling pipeline
    # we only parse text, so no tables/tablestructure , OCR disabled

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,                  
        do_table_structure=False,      
        generate_page_images=False,    
    )

    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print("Parsing PDF with Docling...")
    result = converter.convert(pdf_path)
    doc = result.document

    # Step 2: Run Docling's HybridChunker (first chunking strategy)
    # It:
    #   - Splits at heading boundaries (respects Rule → Section hierarchy)
    #   - Merges short sections that fall below a sensible minimum
    #   - Splits long sections at sentence boundaries if they exceed max_tokens
    #   - Includes the parent heading path in chunk.meta.headings
   
    chunker = HybridChunker(
        tokenizer=TOKENIZER_MODEL,
        max_tokens=MAX_TOKENS,
        merge_peers=True,    # merge tiny adjacent sections under the same heading
    )

    raw_chunks = list(chunker.chunk(doc))
    print(f"Raw chunks from Docling: {len(raw_chunks)}")

    # Step 3: Process each raw chunk into the RulebookChunk format defined earlier

    processed_chunks = []
    skipped_count = 0

    for i, raw_chunk in enumerate(raw_chunks):
        chunk_text = raw_chunk.text.strip()

        # Skip noise chunks
        if is_noise_chunk(chunk_text):
            skipped_count += 1
            continue

        # Get page numbers this chunk spans
        page_numbers = []
        if hasattr(raw_chunk, 'meta') and hasattr(raw_chunk.meta, 'doc_items'):
            for item in raw_chunk.meta.doc_items:
                if hasattr(item, 'prov'):
                    for prov in item.prov:
                        if hasattr(prov, 'page_no'):
                            page_numbers.append(prov.page_no)
        page_numbers = sorted(set(page_numbers))

        # Skip if all pages are in the skip list (defined earlier)
        if page_numbers and all(p in skip_pages for p in page_numbers):
            skipped_count += 1
            continue

        # Extract heading hierarchy from Docling metadata
        headings = []
        if hasattr(raw_chunk, 'meta') and hasattr(raw_chunk.meta, 'headings'):
            headings = raw_chunk.meta.headings or []

        # Parse rule/section info from headings
        heading_info = parse_heading_path(headings)

        # Build context prefix for the chunk text
        context_prefix = build_context_prefix(heading_info, league)

        # Final chunk text = prefix + content
        # The prefix helps BM25 match rule numbers and section names explicitly
        enriched_text = context_prefix + chunk_text

        # Build chunk ID
        chunk_id = f"{doc_id}_chunk_{i:04d}"

        # Build metadata dict (stored alongside the vector in Pinecone)
        metadata = {
            "source": config["source"],
            "layer": 1,
            "layer_name": "rulebook",
            "doc_id": doc_id,
            "league": league,
            "source_url": config["source_url"],
            "chunk_type": heading_info["chunk_type"],
            "rule_number": heading_info["rule_number"],
            "rule_title": heading_info["rule_title"],
            "section": heading_info["section"],
            "section_title": heading_info["section_title"],
            "headings_path": headings,
            "page_numbers": page_numbers,
            "text": enriched_text,
        }

        chunk = RulebookChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=enriched_text,
            metadata=metadata,
        )

        processed_chunks.append(chunk)

    print(f"  Skipped (noise/index): {skipped_count}")
    print(f"  Final usable chunks: {len(processed_chunks)}")

    return processed_chunks

# print a summary of the chunk distribution
def print_chunk_stats(chunks: list[RulebookChunk]) -> None:
    print(f"\n{'='*60}")
    print(f"CHUNK STATISTICS — Total: {len(chunks)}")
    print(f"{'='*60}")

    by_source: dict = {}
    by_type: dict = {}
    token_lengths = []

    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        ctype = chunk.metadata.get("chunk_type", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        by_type[ctype] = by_type.get(ctype, 0) + 1
        # Rough token estimate (words * 1.3)
        token_lengths.append(len(chunk.text.split()) * 1.3)

    print("\nBy source document:")
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count} chunks")

    print("\nBy chunk type:")
    for ctype, count in sorted(by_type.items()):
        print(f"  {ctype}: {count} chunks")

    if token_lengths:
        avg = sum(token_lengths) / len(token_lengths)
        print(f"\nToken length (estimated):")
        print(f"  Average : {avg:.0f} tokens")
        print(f"  Min     : {min(token_lengths):.0f} tokens")
        print(f"  Max     : {max(token_lengths):.0f} tokens")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[RulebookChunk] = []
    for config in RULEBOOKS:
        chunks = process_rulebook(config)
        all_chunks.extend(chunks)

    print_chunk_stats(all_chunks)
    
    # Serialize to JSON
    output_data = [asdict(chunk) for chunk in all_chunks]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(all_chunks)} chunks to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()