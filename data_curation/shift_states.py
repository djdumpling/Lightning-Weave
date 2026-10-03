"""Token-state taxonomy for Qwen3 agent (and reasoning) responses in a Direct-OPD cache.

Each response position t is the state "prefix before response token t". Its
type depends only on that prefix and on the frozen student's cached Top-K
distribution at t, never on the token that was actually sampled, so a label
does not leak the continuation. Types:

- ``think_open``: the first response position (Qwen3 opens ``<think>`` here).
- ``think_body``: ordinary reasoning inside ``<think>``.
- ``think_stop_fork``: inside ``<think>``, the student puts mass on both a
  single-newline sentence end (which precedes ``</think>``) and a paragraph
  break (which continues reasoning). Qwen3 decides to stop thinking here, one
  token before the nearly deterministic ``</think>``.
- ``think_reflection_fork``: the first token of a new reasoning paragraph where
  a reflection marker (Wait, Hmm, But, Alternatively, ...) has mass.
- ``think_close``: ``</think>`` is the likely next token.
- ``act_vs_talk``: after reasoning and outside a call, ``<tool_call>`` competes
  with visible text.
- ``tool_syntax`` / ``tool_name`` / ``tool_args``: inside ``<tool_call>`` JSON.
- ``call_boundary``: right after ``</tool_call>``; end the message or call again.
- ``end_of_message``: outside a call, ``<|im_end|>`` has mass.
- ``post_text``: visible text after reasoning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATE_TYPES = (
    "think_open",
    "think_body",
    "think_stop_fork",
    "think_reflection_fork",
    "think_close",
    "act_vs_talk",
    "tool_syntax",
    "tool_name",
    "tool_args",
    "call_boundary",
    "end_of_message",
    "post_text",
)
STATE_CODE = {name: code for code, name in enumerate(STATE_TYPES)}
REFLECTION_MARKERS = ("Wait", "Hmm", "But", "Alternatively", "Actually", "However", "Hold", "Oh", "Maybe", "Hmm,")
CONCLUSION_MARKERS = ("So", "Okay", "Therefore", "Thus", "Now", "Then")
FORK_MASS = 0.02
NAME_OPEN = re.compile(r'"name"\s*:\s*"([^"\\]|\\.)*$')


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level alphabet (the mapping Qwen's ByteLevel BPE uses)."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    printable += list(range(ord("®"), ord("ÿ") + 1))
    codes = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    return dict(zip(printable, (chr(code) for code in codes)))


_UNICODE_TO_BYTE = {char: byte for byte, char in _bytes_to_unicode().items()}


@dataclass(frozen=True)
class TokenTable:
    """Student token strings, decoded bytes, and structural ids."""

    symbols: dict[int, str]
    added: frozenset[int]
    think: int
    think_close: int
    tool_call: int
    tool_call_close: int
    im_end: int

    @classmethod
    def from_vocab(cls, vocab: dict[str, int], added_tokens: dict[str, int]) -> "TokenTable":
        symbols = {int(token_id): symbol for symbol, token_id in vocab.items()}
        symbols.update({int(token_id): symbol for symbol, token_id in added_tokens.items()})
        ids = {**vocab, **added_tokens}
        return cls(
            symbols=symbols,
            added=frozenset(int(token_id) for token_id in added_tokens.values()),
            think=ids["<think>"],
            think_close=ids["</think>"],
            tool_call=ids["<tool_call>"],
            tool_call_close=ids["</tool_call>"],
            im_end=ids["<|im_end|>"],
        )

    @classmethod
    def from_tokenizer_json(cls, path: str | Path) -> "TokenTable":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        added = {item["content"]: int(item["id"]) for item in data["added_tokens"]}
        return cls.from_vocab(data["model"]["vocab"], added)

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "TokenTable":
        return cls.from_vocab(tokenizer.get_vocab(), tokenizer.get_added_vocab())

    def text(self, token_id: int) -> str:
        """Decoded text of one token; byte fragments of multibyte characters decode leniently."""
        symbol = self.symbols.get(int(token_id), "")
        if int(token_id) in self.added:
            return symbol
        return bytes(_UNICODE_TO_BYTE.get(char, 0x3F) for char in symbol).decode("utf-8", errors="replace")

    def ids_where(self, predicate) -> frozenset[int]:
        return frozenset(token_id for token_id in self.symbols if token_id not in self.added and predicate(self.text(token_id)))


class StateLabeler:
    """Label every response position of a cached row with a :data:`STATE_TYPES` code."""

    def __init__(self, table: TokenTable, *, fork_mass: float = FORK_MASS):
        self.table = table
        self.fork_mass = fork_mass
        self.single_newline = table.ids_where(lambda text: text.endswith("\n") and not text.endswith("\n\n"))
        self.paragraph_break = table.ids_where(lambda text: text.endswith("\n\n"))
        self.reflection = table.ids_where(lambda text: text.strip() in REFLECTION_MARKERS)
        self.conclusion = table.ids_where(lambda text: text.strip() in CONCLUSION_MARKERS)
        size = max(table.symbols) + 2  # the extra last slot is the remaining-vocabulary bucket (id -1)
        self._members = {}
        for name, ids in {
            "single_newline": self.single_newline,
            "paragraph_break": self.paragraph_break,
            "reflection": self.reflection,
            "conclusion": self.conclusion,
            "think_close": {table.think_close},
            "tool_call": {table.tool_call},
            "im_end": {table.im_end},
        }.items():
            member = np.zeros(size, dtype=bool)
            member[list(ids)] = True
            self._members[name] = member

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(self._members)

    def member(self, name: str, token_ids: np.ndarray) -> np.ndarray:
        """Vectorized membership of token ids (id -1 is the remaining-vocabulary bucket, never a member)."""
        return self._members[name][np.asarray(token_ids)]

    def label(self, response_tokens, candidate_ids, behavior_probs, *, prompt_opens_think: bool = False) -> np.ndarray:
        """Return int8 codes [T]. ``behavior_probs`` is [T, K] (full-vocabulary probabilities).

        ``prompt_opens_think`` is for chat templates (Thinking-2507) whose
        generation prompt already ends with ``<think>``.
        """
        table = self.table
        tokens = [int(token) for token in response_tokens]
        candidates = np.asarray(candidate_ids)
        probabilities = np.asarray(behavior_probs, dtype=np.float64)
        mass = {name: (probabilities * self.member(name, candidates)).sum(axis=-1) for name in self._members}
        top = probabilities.max(axis=-1)
        texts = [table.text(token) for token in tokens]
        codes = np.empty(len(tokens), dtype=np.int8)
        in_think, in_call, after_call = prompt_opens_think, False, False
        body = ""
        for position, token in enumerate(tokens):
            previous = texts[position - 1] if position else ""
            if position == 0 and not prompt_opens_think:
                code = "think_open"
            elif in_think:
                if mass["think_close"][position] >= 0.5:
                    code = "think_close"
                elif (
                    mass["single_newline"][position] >= self.fork_mass
                    and mass["paragraph_break"][position] >= self.fork_mass
                ):
                    code = "think_stop_fork"
                elif previous.endswith("\n\n") and mass["reflection"][position] >= self.fork_mass and top[position] < 0.98:
                    code = "think_reflection_fork"
                else:
                    code = "think_body"
            elif in_call:
                code = "tool_name" if NAME_OPEN.search(body) else ("tool_args" if '"arguments"' in body else "tool_syntax")
            elif after_call:
                code = "call_boundary"
            elif mass["tool_call"][position] >= self.fork_mass:
                code = "act_vs_talk"
            elif mass["im_end"][position] >= self.fork_mass:
                code = "end_of_message"
            else:
                code = "post_text"
            codes[position] = STATE_CODE[code]
            # Advance the prefix state with the token that was actually generated.
            if token == table.think:
                in_think = True
            elif token == table.think_close:
                in_think = False
            elif token == table.tool_call:
                in_call, after_call, body = True, False, ""
            elif token == table.tool_call_close:
                in_call, after_call = False, True
            elif in_call:
                body += texts[position]
            elif after_call and texts[position].strip():
                after_call = False
        return codes
