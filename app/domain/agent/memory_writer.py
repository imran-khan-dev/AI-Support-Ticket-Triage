"""Turning a finished agent run into memory drafts.

A run that ended with an answer holds one piece of knowledge that can be
useful later: what the agent told the customer. This module turns that
answer into a memory draft, so that a later run of the same session can
recall it.

The policy is deliberately conservative, because a stored memory becomes
part of the context of the runs that follow it:

* only a successful run produces a memory. A mock answer, a provider
  failure, and a run that ran out of iterations produce nothing;
* only the answer of the agent is used. The customer message, the tool
  calls, the tool results, and the error text are never part of a memory:
  a tool result is documentation, not something that was learned about
  the customer;
* the content is normalized the way the recall side normalizes a memory,
  so that a memory is written and read back the same way;
* an answer that is empty, or that is too short to carry meaning, is
  dropped instead of stored;
* a long answer is shortened at a word boundary, so that it stays a
  readable note and not the fragment of a sentence.

The memory records what the agent said, not a verified fact about the
customer. It is therefore stored with the kind ``summary``.

The module is pure: it does not write to the database, it does not call a
model, and it knows nothing about the HTTP layer. It only returns drafts.
"""

from __future__ import annotations

from app.agent_core.agent import AgentResult
from app.domain.agent.memory import MEMORY_MIN_CONTENT_CHARS
from app.domain.agent.memory import _clean_text as _normalize_text
from app.database.models import MEMORY_KIND_SUMMARY, MemoryDraft


#: Maximum number of characters of the answer that are stored. A memory is
#: a short note, not a transcript.
MEMORY_WRITE_MAX_CHARS = 400


def _shorten(text: str, limit: int) -> str:
    """Return the text, shortened to the limit.

    The text is cut at a word boundary, so that the result does not end in
    the middle of a word. When the cut would leave too little to be
    meaningful, the text is cut at the limit itself.
    """
    if len(text) <= limit:
        return text

    cut = text[:limit]
    space = cut.rfind(" ")
    if space > 0:
        trimmed = cut[:space].rstrip()
        if len(trimmed) >= MEMORY_MIN_CONTENT_CHARS:
            return trimmed
    return cut


def build_memory_drafts(
    result: AgentResult,
    source_msg_id: int | None = None,
) -> list[MemoryDraft]:
    """Return the memory drafts of a finished agent run.

    Args:
        result: The outcome of the run.
        source_msg_id: The id of the stored assistant message the answer
            came from. It is kept as the provenance of the memory.

    Returns:
        A list with one draft when the run produced an answer that is worth
        remembering, and an empty list otherwise. The list is never longer
        than one draft, because a run holds a single answer.
    """
    if result.status != "success":
        return []

    content = _normalize_text(result.answer or "")
    if len(content) < MEMORY_MIN_CONTENT_CHARS:
        return []

    return [
        MemoryDraft(
            kind=MEMORY_KIND_SUMMARY,
            content=_shorten(content, MEMORY_WRITE_MAX_CHARS),
            source_msg_id=source_msg_id,
        )
    ]
