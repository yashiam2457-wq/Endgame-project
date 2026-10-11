"""
Enterprise GraphRAG — Prompts Package.

Consolidated Prompt Templates: Answer Generation, Entity Extraction, Intent Detection, Query Decomposition.
"""

from __future__ import annotations

import sys
from typing import Any

# ── SECTION 1: Answer Generation Prompts ─────────────────────────────────────

ANSWER_GENERATION_SYSTEM = """You are an expert research assistant with access to a curated knowledge base.
Your task is to answer the user's question based ONLY on the provided context.

Rules:
1. Answer based ONLY on the provided context. Do NOT use external knowledge.
2. If the context does not contain sufficient information to answer the question, clearly state:
   "I don't have sufficient information in the provided documents to answer this question."
3. Always cite the specific document chunks you used (e.g., [Doc1-Chunk3]).
4. Be precise, factual, and comprehensive.
5. Use markdown formatting for clarity.
6. Indicate your confidence level at the end: Low / Medium / High.
7. Include a "Reasoning" section explaining how you arrived at your answer.
8. Do NOT fabricate, speculate, or hallucinate facts.

Response format:
## Answer
[Your detailed answer with inline citations like [DocName-ChunkN]]

## Reasoning
[Step-by-step reasoning]

## Supporting Evidence
[Key quotes from the context]

## Confidence
[Low / Medium / High] — [brief justification]"""


def build_answer_prompt(
    query: str,
    context_chunks: list[dict[str, Any]],
    entities: list[str],
) -> str:
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        doc_name = chunk.get("document_filename", f"Doc{i}")
        chunk_idx = chunk.get("sequence_index", i)
        text = chunk.get("text", "")[:1500]
        context_parts.append(
            f"[{doc_name}-Chunk{chunk_idx}]\n{text}"
        )

    context_str = "\n\n---\n\n".join(context_parts)
    entity_str = ", ".join(entities) if entities else "None detected"

    return (
        f"{ANSWER_GENERATION_SYSTEM}\n\n"
        f"DETECTED ENTITIES IN QUERY: {entity_str}\n\n"
        f"RETRIEVED CONTEXT:\n\n{context_str}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"RESPONSE:"
    )


# ── SECTION 2: Entity & Relationship Extraction Prompts ──────────────────────

ENTITY_EXTRACTION_SYSTEM = """Extract named entities and relationships from the text. Output ONLY valid JSON.

Entity types: PERSON, ORGANIZATION, LOCATION, EVENT, PRODUCT, CONCEPT, TECHNOLOGY, DATE, NUMBER, POLICY, LAW, COMPANY, DOCUMENT, DEPENDENCY

Relationship types (UPPERCASE_WITH_UNDERSCORES): USES, IMPLEMENTS, DEPENDS_ON, IMPORTS, CALLS, CREATED_BY, LOCATED_IN, PART_OF, BELONGS_TO, RELATED_TO, MENTIONS, CAUSES, LEADS_TO, REFERENCES, WORKS_WITH, SUPPORTS, COMPATIBLE_WITH, TRAINED_ON, GENERATES, CONSUMES, CONNECTS_TO, HAS_PROPERTY, HAS_VERSION, OWNS, PRODUCES

Output format (JSON only, no other text):
{"entities":[{"name":"Entity Name","type":"TYPE","description":"brief description"}],"relationships":[{"source":"Entity A","target":"Entity B","type":"RELATIONSHIP_TYPE","description":"brief","confidence":0.9,"evidence_sentence":"exact quote from text"}]}"""


def build_entity_extraction_prompt(text: str, max_chars: int = 3000) -> str:
    truncated = text[:max_chars] if len(text) > max_chars else text
    return (
        f"{ENTITY_EXTRACTION_SYSTEM}\n\n"
        f"TEXT:\n{truncated}\n\n"
        f"JSON:"
    )


# ── SECTION 3: Intent Detection Prompts ──────────────────────────────────────

INTENT_DETECTION_PROMPT = """Classify the following query into exactly one intent category.

Intent categories:
- factual: Asking for a specific fact, date, name, or definition
- analytical: Asking for analysis, explanation, or reasoning
- comparative: Asking to compare two or more things
- exploratory: Open-ended exploration of a topic
- navigational: Looking for a specific document or section

Respond with ONLY the intent word, nothing else.

Query: {query}
Intent:"""


def build_intent_prompt(query: str) -> str:
    return INTENT_DETECTION_PROMPT.format(query=query)


# ── SECTION 4: Query Decomposition Prompts ───────────────────────────────────

DECOMPOSITION_PROMPT = """Break down the following complex question into 2-4 simpler, atomic sub-questions.
Each sub-question should be answerable independently and together they should fully address the original question.

Output ONLY a JSON array of strings, for example:
["sub-question 1", "sub-question 2", "sub-question 3"]

Original question: {query}
Sub-questions:"""


def build_decomposition_prompt(query: str) -> str:
    return DECOMPOSITION_PROMPT.format(query=query)


# ── SECTION 5: Backward Compatibility Submodule Aliases ─────────────────────

for _mod in ["answer_generation", "entity_extraction", "intent_detection", "query_decomposition"]:
    sys.modules[f"prompts.{_mod}"] = sys.modules[__name__]
