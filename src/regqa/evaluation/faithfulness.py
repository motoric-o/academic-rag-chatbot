"""NLI entailment faithfulness & hallucination checker.

Owner: Member D (spec §11.2).

Approach (sentence-level NLI):
1. Split the model's answer into individual claims / sentences.
2. For each claim, run an NLI cross-encoder against every provided context
   passage and take the best entailment score.
3. A claim is "supported" if its best entailment score >= threshold.
4. Faithfulness  = fraction of claims that are supported.
5. Hallucination = 1 if any claim is unsupported, else 0 (per-answer flag).

The NLI model (e.g. DeBERTa-v3 MNLI/ANLI) runs on CPU — no GPU required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
#  Lazy-loaded NLI model (heavy import, so only load when first needed)
# ---------------------------------------------------------------------------

_nli_pipeline = None


def _get_nli_pipeline():
    """Load the NLI pipeline lazily on first call (CPU only)."""
    global _nli_pipeline
    if _nli_pipeline is None:
        try:
            from transformers import pipeline  # type: ignore

            _nli_pipeline = pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-base",
                device=-1,  # CPU
            )
        except ImportError:
            raise RuntimeError(
                "transformers is required for the NLI faithfulness checker. "
                "Install with: pip install transformers torch"
            )
    return _nli_pipeline


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClaimResult:
    """NLI result for a single claim."""

    claim: str
    best_label: str  # "ENTAILMENT", "NEUTRAL", "CONTRADICTION"
    best_score: float  # probability of the best_label
    supported: bool  # best entailment score >= threshold
    best_passage_idx: int = -1  # which context passage matched best


@dataclass
class FaithfulnessResult:
    """Aggregated faithfulness result for one model answer."""

    answer: str
    claims: list[ClaimResult] = field(default_factory=list)
    faithfulness_score: float = 0.0  # fraction of supported claims
    hallucination_flag: bool = False  # True if any claim is unsupported
    num_claims: int = 0
    num_supported: int = 0


# ---------------------------------------------------------------------------
#  Sentence splitting
# ---------------------------------------------------------------------------


def split_into_claims(text: str) -> list[str]:
    """Split answer text into atomic claim sentences.

    Uses a simple sentence-boundary regex. For production use, consider
    a proper sentence tokenizer (e.g. spaCy or NLTK).
    """
    # Remove the "Sources: [n], [m]" trailer if present
    text = re.sub(r"\s*Sources:\s*\[.*$", "", text, flags=re.IGNORECASE)
    text = text.strip()
    if not text or text == "NOT_FOUND":
        return []

    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Filter out very short fragments (< 5 chars) and strip
    return [s.strip() for s in sentences if len(s.strip()) >= 5]


# ---------------------------------------------------------------------------
#  Core NLI scoring
# ---------------------------------------------------------------------------


def check_faithfulness(
    answer: str,
    context_passages: list[str],
    threshold: float = 0.5,
    use_model: bool = True,
) -> FaithfulnessResult:
    """Check whether an answer is faithful to the provided context passages.

    Parameters
    ----------
    answer : str
        The model's generated answer text.
    context_passages : list[str]
        The context passages that were fed to the model.
    threshold : float
        Minimum entailment probability to consider a claim supported.
    use_model : bool
        If False, skip NLI model and return a placeholder (for testing).

    Returns
    -------
    FaithfulnessResult
        Aggregated faithfulness metrics and per-claim breakdown.
    """
    claims = split_into_claims(answer)
    result = FaithfulnessResult(answer=answer, num_claims=len(claims))

    if not claims:
        result.faithfulness_score = 1.0  # trivially faithful (no claims)
        return result

    if not context_passages:
        # No context → every claim is unsupported
        result.claims = [
            ClaimResult(
                claim=c,
                best_label="NEUTRAL",
                best_score=0.0,
                supported=False,
            )
            for c in claims
        ]
        result.hallucination_flag = True
        return result

    if not use_model:
        # Placeholder mode: mark all claims as "unknown"
        result.claims = [
            ClaimResult(
                claim=c,
                best_label="NEUTRAL",
                best_score=0.0,
                supported=False,
            )
            for c in claims
        ]
        return result

    # --- Run NLI model ---
    nli = _get_nli_pipeline()
    claim_results: list[ClaimResult] = []

    for claim in claims:
        best_entailment_score = 0.0
        best_label = "NEUTRAL"
        best_passage_idx = -1

        for pidx, passage in enumerate(context_passages):
            # NLI input: premise=passage, hypothesis=claim
            nli_input = f"{passage}</s></s>{claim}"
            preds = nli(nli_input)
            # preds is a list of dicts: [{"label": "ENTAILMENT", "score": 0.95}, ...]
            if isinstance(preds, list) and len(preds) > 0:
                pred = preds[0]
                label = pred.get("label", "NEUTRAL")
                score = pred.get("score", 0.0)

                if label == "ENTAILMENT" and score > best_entailment_score:
                    best_entailment_score = score
                    best_label = label
                    best_passage_idx = pidx

        supported = best_entailment_score >= threshold
        claim_results.append(
            ClaimResult(
                claim=claim,
                best_label=best_label,
                best_score=best_entailment_score,
                supported=supported,
                best_passage_idx=best_passage_idx,
            )
        )

    result.claims = claim_results
    result.num_supported = sum(1 for c in claim_results if c.supported)
    result.faithfulness_score = (
        result.num_supported / result.num_claims if result.num_claims > 0 else 1.0
    )
    result.hallucination_flag = result.num_supported < result.num_claims

    return result


# ---------------------------------------------------------------------------
#  Batch evaluation helper
# ---------------------------------------------------------------------------


def evaluate_faithfulness_batch(
    predictions: list[dict],
    threshold: float = 0.5,
    use_model: bool = True,
) -> dict:
    """Evaluate faithfulness across a batch of predictions.

    Parameters
    ----------
    predictions : list[dict]
        Each dict must have keys: "answer" (str), "context_passages" (list[str]).
        Optionally: "qid" (str) for identification.
    threshold : float
        NLI entailment threshold.
    use_model : bool
        Whether to actually load the NLI model.

    Returns
    -------
    dict
        Aggregate metrics and per-prediction results.
    """
    results: list[dict] = []
    total_faithfulness = 0.0
    total_hallucinated = 0

    for pred in predictions:
        answer = pred.get("answer", pred.get("answer_text", ""))
        passages = pred.get("context_passages", [])
        qid = pred.get("qid", "unknown")

        fr = check_faithfulness(answer, passages, threshold, use_model)

        results.append(
            {
                "qid": qid,
                "faithfulness_score": fr.faithfulness_score,
                "hallucination_flag": fr.hallucination_flag,
                "num_claims": fr.num_claims,
                "num_supported": fr.num_supported,
                "claims": [
                    {
                        "claim": c.claim,
                        "supported": c.supported,
                        "best_label": c.best_label,
                        "best_score": c.best_score,
                        "best_passage_idx": c.best_passage_idx,
                    }
                    for c in fr.claims
                ],
            }
        )
        total_faithfulness += fr.faithfulness_score
        if fr.hallucination_flag:
            total_hallucinated += 1

    n = len(predictions) if predictions else 1
    return {
        "mean_faithfulness": total_faithfulness / n,
        "hallucination_rate": total_hallucinated / n,
        "num_predictions": len(predictions),
        "per_prediction": results,
    }
