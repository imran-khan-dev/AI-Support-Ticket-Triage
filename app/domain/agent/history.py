"""Loading the prior turns of a session for a continuing agent run.

The transcript of a session is stored as rows in the ``messages`` table. This
module turns those rows into the input items the Groq Responses API expects,
so that a run which continues an existing session can see the earlier
conversation.

The loader guarantees:

* **Session isolation.** Only the messages of the requested session are read;
  the session id is part of the query, never a filter applied afterwards.
* **Chronological order.** The rows are read in sequence order, so the items
  are returned oldest first, ready to be placed in front of the new message.
* **Complete tool-call groups.** A tool call and its output are replayed
  together, and a group is never cut in half by the size limit. An output
  without its call, and a call without its output, are left out, because the
  provider cannot match half a pair.
* **Distinct call ids.** A replayed call gets a call id of its own, so it can
  never be confused with a call the model makes in the current turn.
* **No duplication of the current message.** The loader only returns what was
  stored earlier. The message of the current run is added once, by the agent
  loop, after the history.

The module knows nothing about the provider or the HTTP layer; it only
produces plain dictionaries.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from app.core.config import DEFAULT_MAX_HISTORY_MESSAGES
from app.database.db import Database
from app.database.models import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    Message,
)
from app.database.repository import list_messages


def _user_item(message: Message) -> dict[str, Any]:
    """Return the input item of a stored user message."""
    return {
        "id": f"hist_msg_{message.id}",
        "type": "message",
        "role": ROLE_USER,
        "content": [{"type": "input_text", "text": message.content}],
    }


def _assistant_item(message: Message) -> dict[str, Any]:
    """Return the input item of a stored assistant answer."""
    return {
        "id": f"hist_msg_{message.id}",
        "type": "message",
        "role": ROLE_ASSISTANT,
        "content": [{"type": "output_text", "text": message.content}],
    }


def _tool_name_of(message: Message) -> str | None:
    """Return the tool a stored record belongs to, or ``None`` when unknown.

    The tool name is kept in its own column. The content of an assistant
    tool-call record carries the same name as a fallback, so a record that
    was written without the column can still be replayed.
    """
    if message.tool_name:
        return message.tool_name
    try:
        payload = json.loads(message.content)
    except (TypeError, ValueError):
        return None
    if isinstance(payload, dict):
        name = payload.get("tool")
        if isinstance(name, str) and name:
            return name
    return None


def _function_call_item(
    message: Message,
    name: str,
    call_id: str,
) -> dict[str, Any]:
    """Return the input item of a stored assistant tool call.

    The Responses API carries the arguments of a call as a JSON string, so
    the stored arguments are serialized again.
    """
    arguments: Any = {}
    try:
        payload = json.loads(message.content)
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        arguments = payload.get("arguments") or {}
    return {
        "id": f"hist_fc_{message.id}",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments, sort_keys=True, default=str),
    }


def _function_call_output_item(
    message: Message,
    name: str,
    call_id: str,
) -> dict[str, Any]:
    """Return the input item of a stored tool result."""
    return {
        "id": f"hist_output_{message.id}",
        "type": "function_call_output",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "output": message.content,
    }


def _group_messages(messages: Sequence[Message]) -> list[list[Message]]:
    """Group the transcript into units that have to be replayed together.

    A tool call and its output form one unit. A tool output whose call is
    missing is left out, and so is a call whose output never arrived: half a
    pair cannot be matched by the provider, and Groq's Harmony renderer
    rejects such a request.
    """
    groups: list[list[Message]] = []
    open_calls: dict[str, list[Message]] = {}
    for message in messages:
        if message.role == ROLE_ASSISTANT and message.call_id:
            group = [message]
            groups.append(group)
            open_calls[message.call_id] = group
        elif message.role == ROLE_TOOL and message.call_id:
            group = open_calls.pop(message.call_id, None)
            if group is None:
                continue
            group.append(message)
        else:
            groups.append([message])
    return [
        group
        for group in groups
        if not (
            len(group) == 1
            and group[0].role == ROLE_ASSISTANT
            and group[0].call_id
        )
    ]


def _group_items(group: Sequence[Message]) -> list[dict[str, Any]]:
    """Return the input items of one group of stored records."""
    call = group[0]
    if call.role == ROLE_ASSISTANT and call.call_id:
        name = _tool_name_of(call)
        if name is None:
            return []
        # The call id of a replayed group is derived from the stored row, so
        # it can never be equal to the call id of a call the model makes in
        # the current turn. The output keeps the id of its call.
        call_id = f"hist_call_{call.id}"
        items = [_function_call_item(call, name, call_id)]
        for message in group[1:]:
            items.append(
                _function_call_output_item(
                    message, message.tool_name or name, call_id
                )
            )
        return items

    message = call
    if message.role == ROLE_USER:
        return [_user_item(message)]
    if message.role == ROLE_ASSISTANT:
        return [_assistant_item(message)]
    # System records hold the safe error message of a failed run. They are
    # not part of the conversation and are not replayed into the model.
    return []


def _trim_groups(
    groups: Sequence[Sequence[dict[str, Any]]],
    max_messages: int,
) -> list[list[dict[str, Any]]]:
    """Keep the most recent groups that fit within the limit.

    A group is never split, so a tool call and its output are either both
    replayed or both left out. The result can therefore be shorter than the
    limit.
    """
    if max_messages <= 0:
        return []
    kept: list[list[dict[str, Any]]] = []
    total = 0
    for group in reversed(groups):
        size = len(group)
        if total + size > max_messages:
            break
        kept.append(group)
        total += size
    kept.reverse()
    return kept

# Main Function
def load_history_items(
    db: Database,
    session_id: str,
    *,
    max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    exclude_correlation_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the prior turns of a session as Responses API input items.

    Args:
        db: The database handle.
        session_id: The session whose transcript is loaded. Only the
            messages of this session are returned.
        max_messages: The maximum number of items to replay. The limit is
            applied to whole groups, so a tool call is never separated from
            its output.
        exclude_correlation_ids: Correlation ids of runs that must not be
            replayed. The caller passes the id of the current run, so the
            message of the current run is never sent twice.

    Returns:
        The input items of the stored transcript, oldest first. The message
        of the current run is not part of them: the agent loop appends it.
    """
    if max_messages <= 0:
        return []

    excluded = {
        str(value) for value in (exclude_correlation_ids or ()) if value
    }

    # One message more than the limit is read, so that a group the window
    # cut in half can still be dropped as a whole instead of being replayed
    # half.
    messages = list_messages(db, session_id, limit=max_messages + 1)
    if excluded:
        messages = [
            message
            for message in messages
            if message.correlation_id not in excluded
        ]

    groups = _group_messages(messages)
    item_groups = [
        items for items in (_group_items(group) for group in groups) if items
    ]
    kept = _trim_groups(item_groups, max_messages)
    return [item for group in kept for item in group]
