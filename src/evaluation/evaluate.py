import json
import os
import re
import time
import textwrap
from dataclasses import dataclass, field
from typing import Optional # for static type hints

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JUDGE_MODEL_ID = "gemma-3-4b-it"
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

# Sliding-window rate limiter for the Gemini API
class RateLimiter:
    """
    Tracks API calls and token usage in a rolling 60-second window and
    sleeps the minimum required time before any call that would breach
    either limit.

    Why sliding window instead of a fixed "reset every minute" bucket?
    A fixed bucket is fragile — if you fire 14 calls at t=0:59 and 14
    more at t=1:01 you've used 28 calls in 2 seconds while technically
    staying inside two separate minute buckets.  A sliding window always
    looks at the last 60 seconds, so the constraint is continuously
    enforced regardless of where you are in the clock.

    Gemini free tier hard limits: 30 RPM, 15 000 TPM.
    We set effective limits slightly below to absorb timing jitter.
    """
    EFFECTIVE_RPM: int   = 28        # hard limit: 30
    EFFECTIVE_TPM: int   = 14_000    # hard limit: 15 000
    WINDOW:        float = 60.0      # seconds

    def __init__(self) -> None:
        # Each entry: (unix_timestamp_of_call, tokens_consumed_by_that_call)
        self._history: list[tuple[float, int]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Drop entries older than WINDOW seconds."""
        cutoff = time.time() - self.WINDOW
        self._history = [(t, tok) for t, tok in self._history if t > cutoff]

    def _window_stats(self) -> tuple[int, int]:
        """Return (call_count, token_count) within the current window."""
        self._prune()
        calls  = len(self._history)
        tokens = sum(tok for _, tok in self._history)
        return calls, tokens

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def wait_if_needed(self, estimated_tokens: int) -> None:
        """
        Block until there is headroom for one more call that will consume
        approximately *estimated_tokens* tokens.

        Called BEFORE every _llm() invocation.
        """
        while True:
            calls, tokens = self._window_stats()

            rpm_ok = calls  < self.EFFECTIVE_RPM
            tpm_ok = tokens + estimated_tokens <= self.EFFECTIVE_TPM

            if rpm_ok and tpm_ok:
                return   # safe to proceed

            # Sleep until the oldest entry slides out of the window.
            # Adding a 300 ms buffer avoids immediately re-entering the loop
            # due to floating-point timing imprecision.
            if self._history:
                oldest_ts  = self._history[0][0]
                sleep_secs = (oldest_ts + self.WINDOW) - time.time() + 0.3
                sleep_secs = max(0.3, sleep_secs)
            else:
                sleep_secs = 1.0

            print(f"[RateLimiter] Window full ({calls} calls, {tokens} tokens). "
                  f"Sleeping {sleep_secs:.1f}s...")
            time.sleep(sleep_secs)

    def record(self, tokens_used: int) -> None:
        """
        Record a completed call.
        Called AFTER every _llm() invocation with the actual token count
        from the API response (or the pre-call estimate if the response
        did not include usage data).
        """
        self._history.append((time.time(), tokens_used))

    @staticmethod
    def estimate_tokens(prompt: str, max_response_tokens: int) -> int:
        """
        Rough upper-bound estimate of total tokens for a call.
        Rule of thumb: ~4 chars per token for English text.
        We add max_response_tokens to account for the output side of the
        TPM budget, since Gemini counts both input and output tokens.
        """
        input_estimate = max(1, len(prompt) // 4)
        return input_estimate + max_response_tokens


class RAGJudge:

    def __init__(self, verbose: bool = True):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found.")
        self.verbose = verbose
        self._log("Loading judge LLM client...")
        self.client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GOOGLE_API_KEY,
        )
        self._log(f"Loading sentence similarity model: {SIM_MODEL_NAME}...")
        self.sim_model = SentenceTransformer(SIM_MODEL_NAME)

        # One shared rate limiter for all LLM calls made by this judge instance
        self.rate_limiter = RateLimiter()

        self._log("Judge ready.")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Judge] {msg}")

    def _llm(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """
        Single-turn LLM call with integrated rate limiting.

        Flow:
          1. Estimate token cost of this call (prompt tokens + max response tokens).
          2. Ask the rate limiter to sleep if the window is too full.
          3. Make the API call.
          4. Record the actual token count (from response.usage if present,
             otherwise fall back to our pre-call estimate).

        temperature=0.0 for deterministic judge outputs.
        System role not supported by Gemma chat template — merged into user turn.
        """
        estimated_tokens = RateLimiter.estimate_tokens(prompt, max_tokens)

        # Block here if needed — this is the only sleep in the whole eval loop
        self.rate_limiter.wait_if_needed(estimated_tokens)

        try:
            response = self.client.chat.completions.create(
                model=JUDGE_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )

            # Use actual token count when the API returns it; fall back to estimate.
            # Gemini's OpenAI-compat endpoint populates response.usage reliably.
            actual_tokens = (
                response.usage.total_tokens
                if response.usage and response.usage.total_tokens
                else estimated_tokens
            )
            self.rate_limiter.record(actual_tokens)

            return (response.choices[0].message.content or "").strip()

        except Exception as e:
            # Still record estimated usage so the limiter doesn't think the
            # window is emptier than it really is after a failed call.
            self.rate_limiter.record(estimated_tokens)
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
            You are a reasonable and accurate fact-checker.

            CONTEXT (retrieved from a basketball rulebook / encyclopedia):
            {context}

            CLAIM TO VERIFY:
            {claim}

            TASK: Does the CONTEXT support this claim?
            Rules:
            - Answer "SUPPORTED" if the claim is explicitly stated OR reasonably implied by the context.
            - Allow for synonyms and paraphrasing (e.g., if the text says someone is a "post player", it supports the claim that they "play in the post").
            - Answer "NOT SUPPORTED" only if the claim is entirely missing, directly contradicted, or requires outside knowledge.
            # - Do not use your own basketball knowledge — judge only from the CONTEXT above.

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

    def faithfulness(self, answer: str, context_chunks: list[dict]) -> FaithfulnessResult:
        """
        Full faithfulness pipeline for one query.
        context_chunks: the list of retrieved chunk dicts

        The old time.sleep(1) between claim verifications has been removed —
        the rate limiter in _llm() now handles all pacing adaptively.
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
            # No sleep here — _llm() calls wait_if_needed() before every call
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
    def evaluate(self, query: str, answer: str, context_chunks: list[dict]) -> EvalResult:
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

TEST_QUERIES = [
    # NBA rules
    "What is the defensive three-second rule in the NBA?",
    "What is the 'Restricted Area' on an NBA court and how far is it from the basket?",
    "What are the rules for a technical foul in the NBA?",
    "How does the shot clock reset work after an offensive rebound in the NBA?",
    # FIBA rules
    "What is the rule for a backcourt violation in FIBA?",
    "How many seconds does a player have to inbound the ball in FIBA?",
    # Strategy / tactics
    "Coach's Challenge: We have a very athletic team but we are undersized. Which press defense (1-2-1-1 or 1-2-2) should we use to maximize our speed for turnovers?",
    "How does a zone defense differ from man-to-man defense?",
    "What is the purpose of a box-and-one defense?",
    "Player Scenario: I'm guarding a 'slasher' in a 1v1 game. Should I play 'tight pressure' or 'sagging defense', and why?",
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