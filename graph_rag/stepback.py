"""
Step-back prompting: rewrite a specific question into a broader one for better vector retrieval.
Used only for chunk retrieval (vector search); the original question is used for fulltext and for the final answer.
"""
from __future__ import annotations

import logging

from graph_rag.llm import generate

log = logging.getLogger(__name__)

STEP_BACK_PROMPT = """You are an expert at formulating broader, more general questions that help find relevant context.

Given the following specific question, produce a single step-back question that is more general and would help retrieve documents containing the information needed to answer the original question.

Examples:
- Specific: "Which team did Thierry Audel play for from 2007 to 2008?" → Step-back: "Which teams did Thierry Audel play for in his career?"
- Specific: "What is the deadline in section 50?" → Step-back: "What does section 50 say about deadlines and procedures?"

Respond with only the step-back question, nothing else. No quotes or explanation.

Specific question:
{question}

Step-back question:"""


def generate_step_back_question(question: str) -> str:
    """Generate a broader step-back question from the given specific question."""
    prompt = STEP_BACK_PROMPT.format(question=question.strip())
    out = generate(prompt).strip()
    if out:
        print("Step-back question (vector retrieval):", out, flush=True)
        log.info("Step-back question (for vector retrieval): %s", out[:200] + ("…" if len(out) > 200 else ""))
    return out if out else question
