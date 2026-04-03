import os
import textwrap
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
# from huggingface_hub import InferenceClient
from openai import OpenAI

load_dotenv()
# HF_TOKEN      = os.getenv("HF_TOKEN")   # set in .env locally, HF Spaces Secret in prod
# MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.3" # unfortunatley not available anymore freely

# --- UPDATED SECTION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_ID     = "llama-3.1-8b-instant"  # Groq's model ID
# -----------------------

MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.2    
TOP_P           = 0.9
# REPETITION_PENALTY = 1.15  # mild penalty to stop the model looping on rule text

# How many retrieved chunks to include in the context window
MAX_CONTEXT_CHUNKS = 5

@dataclass
class GenerationResult:
    """
    for use in both the UI and evaluation
    answer:         The LLM's generated answer string
    query:          The original user question (passed through for convenience)
    context_chunks: The retrieved chunks that were fed into the prompt
                    Each dict has: chunk_id, text, metadata, cross_encoder_score, rrf_score
    prompt:         The full prompt sent to the LLM (useful for debugging + report)
    model_id:       Which model produced the answer
    """
    answer:         str
    query:          str
    context_chunks: list[dict]  = field(default_factory=list)
    prompt:         str         = ""
    model_id:       str         = MODEL_ID

# next two functions help in building the prompt
def _format_chunk_for_context(chunk: dict, index: int) -> str:
    # Formats one retrieved chunk into a labelled context block, constructed to be able to cite sources
    
    meta       = chunk.get("metadata", {})
    layer_name = meta.get("layer_name", "")
    league     = meta.get("league", "")
    term       = meta.get("term", "")
    # rule       = meta.get("rule_number", "")
    # rule_title = meta.get("rule_title", "")
    section    = meta.get("section", "")
    sec_title  = meta.get("section_title", "")

    # Build a human-readable source label
    if layer_name == "rulebook":
        label_parts = [p for p in [league, "Rulebook"] if p]
        # if rule and rule_title:
        #     label_parts.append(f"{rule} ({rule_title})")
        # rule_number and rule_title is null in the chunks right now, so skip for now
        if section and sec_title:
            label_parts.append(f"{section} ({sec_title})")
        label = " | ".join(label_parts)
    else:
        label_parts = ["HoopStudent Basketball Encyclopedia"]
        if term:
            label_parts.append(term)
        chunk_type = meta.get("chunk_type", "")
        if chunk_type:
            label_parts.append(chunk_type.title())
        label = " | ".join(label_parts)

    text = chunk.get("text", "").strip()

    return f"[Source {index} — {label}]\n{text}"


def build_prompt(query: str, context_chunks: list[dict]) -> tuple[str, str]:
    """
    Builds the full  prompt.

    Prompt design decisions:
      - System block sets a strict "only use the provided context" rule.
        This directly controls faithfulness — the LLM should not draw on its
        basketball knowledge beyond what's in the chunks.
      - We tell it to cite sources by [Source N] so the app can surface them.
      - We ask it to acknowledge uncertainty rather than hallucinate.
      - The context is numbered so cross-referencing is unambiguous.
      - "concise but complete" nudges against both truncation and rambling.
    """
    # Format all context chunks
    context_blocks = "\n\n".join(
        _format_chunk_for_context(chunk, i + 1)
        for i, chunk in enumerate(context_chunks)
    )
    # modified prmpt
    system_prompt = textwrap.dedent("""
        You are an expert, educational basketball rules and strategy assistant.
        Your job is to answer questions about basketball rules (NBA and FIBA) and basketball tactics based ONLY on the provided context

        STRICT RULES:
        1. GROUNDING: Answer ONLY using the provided context sources. Do NOT use outside knowledge.
        2. CITATIONS: Whenever you state a fact from the context, cite the source(s) using [Source N] inline.
        3. UNKNOWN: If the context does not contain enough information to answer, say:
           "I could not find a definitive answer in the available sources."
           Do NOT guess or fabricate rule numbers, penalties, or tactical details.
        4. FORMATTING AND READIBILITY (CRITICAL):
           - Structure your answer logically.
           - Use short paragraphs and line breaks to separate distinct ideas. 
           - Use bullet points if listing multiple rules, steps, or differences.
           - Do NOT output a single massive wall of text.                            
        5. TONE: Be explanatory and easy to understand. Synthesize the context naturally instead of awkwardly stringing quotes together.
    """).strip()

    user_message = textwrap.dedent(f"""
        CONTEXT SOURCES:
        {context_blocks}

        ---

        QUESTION: {query}

        Using only the context sources above, provide a clear and accurate answer.
        Cite your sources inline using [Source N].
    """).strip()

    return system_prompt, user_message

class BasketballGenerator:
    """
    Wraps the InferenceClient and exposes a single .generate() method.

    InferenceClient is preferred over raw requests because:
      - Handles auth headers automatically
      - Supports chat_completion format (system + user messages)
      - Retries on transient 503s (model loading)
      - Works identically locally and on HF Spaces
    """
    def __init__(self, model_id: str = MODEL_ID, verbose: bool = True):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Set it in .env locally or as a Space secret on HuggingFace."
            )
        self.model_id = model_id
        self.verbose  = verbose
 
        # OpenRouter is OpenAI-API-compatible — just swap the base_url
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
        )
        self._log(f"Generator ready — model: {model_id} via Groq")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Generator] {msg}")

    def generate(self, query: str,retrieved_chunks: list[dict],
        max_chunks:     int   = MAX_CONTEXT_CHUNKS,
        max_new_tokens: int   = MAX_NEW_TOKENS,
        temperature:    float = TEMPERATURE,
        top_p:          float = TOP_P,
    ) -> GenerationResult:
        
        # Generates an answer grounded in the retrieved chunks.
    
        
        # Use only the top-N chunks by CrossEncoder score
        context_chunks = retrieved_chunks[:max_chunks]

        if not context_chunks:
            return GenerationResult(
                answer="No relevant context was retrieved. Please try rephrasing your question.",
                query=query,
                context_chunks=[],
                prompt="",
                model_id=self.model_id,
            )

        self._log(f"Building prompt with {len(context_chunks)} context chunks...")
        system_prompt, user_message = build_prompt(query, context_chunks)

        # Full prompt string for logging/report
        full_prompt_log = f"[SYSTEM+USER MERGED]\n{system_prompt}\n\n{user_message}"

        self._log(f"Calling {self.model_id}...")

        try:
            # chat_completion handles the templating automatically
            # for gemma models, system role isn't supported in chat template, so we merge it with the user message
            response = self.client.chat.completions.create(
                model=self.model_id,
                # gemma needed this, for llama I can separate it
                # messages=[ 
                # {
                # "role": "user",
                # "content": f"{system_prompt}\n\n{user_message}",
                # },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,)
            content = response.choices[0].message.content
            answer = content.strip() if content else "No response generated."
            self._log(f"Generated {len(answer.split())} words.")

        except Exception as e:
            # Surface the error clearly rather than silently returning empty string
            error_msg = str(e)
            self._log(f"Inference API error: {error_msg}")
            # Common failure modes and human-readable explanations:
            if "loading" in error_msg.lower() or "503" in error_msg:
                answer = (
                    "The language model is currently loading on Hugging Face servers. "
                    "Please wait 20–30 seconds and try again."
                )
            elif "quota" in error_msg.lower() or "429" in error_msg:
                answer = (
                    "API rate limit reached. "
                    "Please wait a moment before submitting another query."
                )
            else:
                answer = f"Generation failed: {error_msg}"

        return GenerationResult(
            answer=answer,
            query=query,
            context_chunks=context_chunks,
            prompt=full_prompt_log,
            model_id=self.model_id,
        )
# Convenience function — import this in app.py and judge.py
_generator_singleton: Optional[BasketballGenerator] = None
def get_generator(model_id: str = MODEL_ID) -> BasketballGenerator:
    """
    Returns a cached generator instance.
    Prevents re-initialising the InferenceClient on every request.
    """
    global _generator_singleton
    if _generator_singleton is None:
        _generator_singleton = BasketballGenerator(model_id=model_id)
    return _generator_singleton


# ---------------------------------------------------------------------------
# Smoke test — run:  python src/generation/generator.py
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     from src.retrieval.retriever import get_retriever

#     retriever = get_retriever()
#     generator = get_generator()

#     TEST_QUERIES = [
#         "What is the NBA rule on defensive three seconds?",
#         "Explain how a pick and roll works.",
#         "What is the shot clock duration in FIBA rules?",
#     ]

#     for query in TEST_QUERIES:
#         print("\n" + "=" * 60)
#         print(f"QUERY: {query}")
#         print("=" * 60)

#         chunks = retriever.retrieve(query, final_n=5)
#         result = generator.generate(query, chunks)

#         print(f"\nANSWER:\n{result.answer}")
#         print(f"\nSOURCES USED ({len(result.context_chunks)}):")
#         for i, c in enumerate(result.context_chunks, 1):
#             meta = c.get("metadata", {})
#             label = meta.get("rule_number") or meta.get("term") or c.get("chunk_id")
#             print(f"  [{i}] {label}  (CE={c.get('cross_encoder_score', 0):+.3f})")