import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional # for static type hints

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
JUDGE_MODEL_ID     = "meta-llama/llama-3.2-3b-instruct:free" #using the smaller gemma model for evaluation

# Lightweight bi-encoder for relevancy cosine similarity.
# all-MiniLM-L6-v2 is 80MB, fast on CPU, well-suited for short sentence similarity.
SIM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class ClaimVerification:
    claim:     str
    supported: bool
    reasoning: str = ""   # brief LLM explanation


@dataclass
class FaithfulnessResult:
    score:   float              # 0–1: fraction of claims supported
    claims:  list[ClaimVerification] = field(default_factory=list)
    total:   int = 0
    supported_count: int = 0


@dataclass
class RelevancyResult:
    score:              float         # 0–1: mean cosine sim of 3 generated Qs vs original Q
    generated_questions: list[str]   = field(default_factory=list)
    similarities:        list[float] = field(default_factory=list)


@dataclass
class EvalResult:
    query:       str
    answer:      str
    faithfulness: FaithfulnessResult
    relevancy:    RelevancyResult

    @property
    def faithfulness_score(self) -> float:
        return round(self.faithfulness.score, 4)

    @property
    def relevancy_score(self) -> float:
        return round(self.relevancy.score, 4)

class RAGJudge:

    def __init__(self, verbose: bool = True):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found. ")
        self.verbose = verbose
        self._log("Loading judge LLM client...")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        self._log(f"Loading sentence similarity model: {SIM_MODEL_NAME}...")
        self.sim_model = SentenceTransformer(SIM_MODEL_NAME)
        self._log("Judge ready.")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Judge] {msg}")

    def _llm(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """
        Single-turn LLM call. temperature=0.0 for deterministic judge outputs.
        System role not supported by Gemma chat template — merged into user turn.
        """
        try:
            response = self.client.chat.completions.create(
                model=JUDGE_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            self._log(f"LLM call failed: {e}")
            return ""

    def _extract_claims(self, answer: str) -> list[str]:
        """
        Step 1 of faithfulness: extract atomic factual claims from the answer, in JSON output
        """
        prompt = textwrap.dedent(f"""
            You are a precise fact-extractor.

            Read the following answer and extract every distinct factual claim it makes.
            A claim is a short declarative sentence that asserts one specific fact.
            Do NOT include opinions, hedges ("may", "could"), or meta-statements.

            Return ONLY a valid JSON array of strings. No preamble, no explanation.
            Example output: ["The shot clock is 24 seconds in the NBA.", "A player must release the ball before the buzzer sounds."]

            ANSWER:
            {answer}

            JSON array of claims:
        """).strip()

        raw = self._llm(prompt, max_tokens=400)

        # Parse JSON, falling back gracefully
        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            claims = json.loads(clean)
            if isinstance(claims, list):
                return [str(c).strip() for c in claims if str(c).strip()]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: split on newlines / numbered lines
        lines = [re.sub(r"^\s*[\d\.\-\*]+\s*", "", line).strip() for line in raw.split("\n")]
        return [l for l in lines if len(l) > 15]

    def _verify_claim(self, claim: str, context: str) -> ClaimVerification:
        """
        Step 2 of faithfulness: verify one claim against the retrieved context.
        Returns a ClaimVerification with supported=True/False + a short reasoning.
        """
        prompt = textwrap.dedent(f"""
            You are a strict fact-checker.

            CONTEXT (retrieved from a basketball rulebook / encyclopedia):
            {context}

            CLAIM TO VERIFY:
            {claim}

            Does the CONTEXT directly support this claim?
            Rules:
            - Answer "SUPPORTED" only if the context explicitly states or clearly implies the claim.
            - Answer "NOT SUPPORTED" if the claim is absent, contradicted, or requires outside knowledge.
            - Do not use your own basketball knowledge — judge only from the CONTEXT above.

            Respond in this exact format (two lines):
            VERDICT: SUPPORTED   or   VERDICT: NOT SUPPORTED
            REASON: <one sentence explaining why>
        """).strip()

        raw = self._llm(prompt, max_tokens=120)

        verdict_match = re.search(r"VERDICT:\s*(SUPPORTED|NOT SUPPORTED)", raw, re.IGNORECASE)
        reason_match  = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE)

        supported = bool(verdict_match and "NOT" not in verdict_match.group(1).upper())
        reasoning = reason_match.group(1).strip() if reason_match else raw[:120]

        return ClaimVerification(claim=claim, supported=supported, reasoning=reasoning)

    def faithfulness(self,answer: str,context_chunks: list[dict],) -> FaithfulnessResult:
        """
        Full faithfulness pipeline for one query.
        context_chunks: the list of retrieved chunk dicts 
        """
        # Concatenate all chunk texts into one context string for the verifier
        context_text = "\n\n---\n\n".join(
            c.get("text", "") for c in context_chunks if c.get("text")
        )

        self._log("Extracting claims from answer...")
        claims = self._extract_claims(answer)
        self._log(f"  Found {len(claims)} claims.")

        if not claims:
            return FaithfulnessResult(score=0.0, claims=[], total=0, supported_count=0)

        verifications = []
        for i, claim in enumerate(claims):
            self._log(f"  Verifying claim {i+1}/{len(claims)}: {claim[:60]}...")
            result = self._verify_claim(claim, context_text)
            time.sleep(1) # second pause helps avoid hitting the "burst" limit
            verifications.append(result)

        supported = sum(1 for v in verifications if v.supported)
        score = supported / len(verifications)

        return FaithfulnessResult(
            score=score,
            claims=verifications,
            total=len(verifications),
            supported_count=supported,
        )

    def _generate_questions(self, answer: str) -> list[str]:
        """
        Step 1 of relevancy: generate 3 questions that the answer could be answering.
        These are then compared (via cosine similarity) to the original query.
        """
        prompt = textwrap.dedent(f"""
            You are a question generator.

            Read the following answer about basketball rules or strategy.
            Generate exactly 3 different questions that this answer could be a response to.
            The questions should be specific, natural, and varied.

            Return ONLY a valid JSON array of exactly 3 question strings. No preamble.
            Example: ["What is the shot clock rule in the NBA?", "How long does a team have to shoot?", "What happens if the shot clock expires?"]

            ANSWER:
            {answer}

            JSON array of 3 questions:
        """).strip()

        raw = self._llm(prompt, max_tokens=200)

        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            questions = json.loads(clean)
            if isinstance(questions, list):
                qs = [str(q).strip() for q in questions if str(q).strip()]
                return qs[:3]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: line-by-line
        lines = [re.sub(r"^\s*[\d\.\-\*]+\s*", "", line).strip() for line in raw.split("\n")]
        qs = [l for l in lines if len(l) > 10 and "?" in l]
        return qs[:3]

    def relevancy(self, query: str, answer: str) -> RelevancyResult:
        """
        Full relevancy pipeline for one query.
        Generates 3 questions from the answer, then measures cosine similarity
        between each generated question and the original query.
        """
        self._log("Generating alternate questions for relevancy...")
        questions = self._generate_questions(answer)
        self._log(f"  Generated {len(questions)} questions.")

        if not questions:
            return RelevancyResult(score=0.0, generated_questions=[], similarities=[])

        # Embed original query + all generated questions in one batch
        all_texts   = [query] + questions
        embeddings  = self.sim_model.encode(all_texts, normalize_embeddings=True)

        query_vec   = embeddings[0]           # shape (384,)
        question_vecs = embeddings[1:]        # shape (n, 384)

        # Cosine similarity — since embeddings are L2-normalised, sim = dot product
        similarities = [float(query_vec @ qv) for qv in question_vecs]
        mean_score   = float(np.mean(similarities)) if similarities else 0.0

        return RelevancyResult(
            score=mean_score,
            generated_questions=questions,
            similarities=similarities,
        )

    # Combining it all:
    def evaluate(self,query: str,answer:  str, context_chunks: list[dict],) -> EvalResult:
        self._log(f"\nEvaluating: {query[:60]}...")
        faith = self.faithfulness(answer, context_chunks)
        rel   = self.relevancy(query, answer)

        self._log(
            f"  Faithfulness: {faith.score:.2%} ({faith.supported_count}/{faith.total} claims supported)"
        )
        self._log(f"  Relevancy:    {rel.score:.4f} (mean cosine sim)")

        return EvalResult(
            query=query,
            answer=answer,
            faithfulness=faith,
            relevancy=rel,
        )

# Fixed test set — 15 basketball queries covering NBA, FIBA, and strategy
TEST_QUERIES = [
    # NBA rules
    # "What is the defensive three-second rule in the NBA?",
    # "How many personal fouls does it take to foul out in the NBA?",
    # "What is the NBA rule for a player being out of bounds?",
    # "What are the rules for a technical foul in the NBA?",
    # "How does the shot clock reset work after an offensive rebound in the NBA?",
    # # FIBA rules
    # "What is the shot clock duration in FIBA basketball?",
    # "How is a jump ball situation resolved in FIBA rules?",
    # "What constitutes a legal screen according to FIBA rules?",
    # "What is the rule for a backcourt violation in FIBA?",
    # "How many seconds does a player have to inbound the ball in FIBA?",
    # # Strategy / tactics
    # "What is a pick and roll and how does it work?",
    # "How does a zone defense differ from man-to-man defense?",
    "What is the purpose of a box-and-one defense?",
    "How does an isolation play work in basketball offense?",
    "What is a Princeton offense and what are its key principles?",
]

_judge_singleton: Optional[RAGJudge] = None

def get_judge() -> RAGJudge:
    global _judge_singleton
    if _judge_singleton is None:
        _judge_singleton = RAGJudge(verbose=True)
    return _judge_singleton


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     judge = RAGJudge(verbose=True)

#     test_answer = (
#         "In the NBA, a player in the paint on defense cannot remain there for "
#         "more than three consecutive seconds while their team is in control of "
#         "the ball [Source 1]. This rule is called the defensive three-second "
#         "violation and results in a technical foul [Source 2]."
#     )
#     test_query = "What is the defensive three seconds rule in the NBA?"
#     test_chunks = [
#         {"text": "NBA Rule No. 26—Defensive Three-Second Rule: No defensive player may "
#                  "remain in the lane for more than three consecutive seconds while the "
#                  "ball is in play. Violation results in a technical foul."}
#     ]

#     result = judge.evaluate(test_query, test_answer, test_chunks)
#     print(f"\nFaithfulness: {result.faithfulness_score:.2%}")
#     for cv in result.faithfulness.claims:
#         mark = "✓" if cv.supported else "✗"
#         print(f"  {mark} {cv.claim}")

#     print(f"\nRelevancy: {result.relevancy_score:.4f}")
#     for q, s in zip(result.relevancy.generated_questions, result.relevancy.similarities):
#         print(f"  {s:.4f} — {q}")