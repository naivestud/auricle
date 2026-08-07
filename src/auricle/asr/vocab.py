"""Character-level vocabulary for CTC decoding."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

CHARS = tuple("abcdefghijklmnopqrstuvwxyz' ")
"""The default character set: lowercase letters, apostrophe and space."""


class CharVocabulary:
    """Character vocabulary with a CTC blank token at index 0.

    Token layout: ``[blank, <char 1>, <char 2>, ...]`` so token ids are
    ``1 + position in chars``.
    """

    BLANK = 0

    def __init__(self, chars: Sequence[str] = CHARS):
        self.chars = tuple(chars)
        self._char_to_id = {c: i + 1 for i, c in enumerate(self.chars)}
        self._id_to_char = {i + 1: c for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars) + 1

    @property
    def size(self) -> int:
        return len(self)

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids.

        Raises ``ValueError`` on characters outside the vocabulary.
        """
        ids: list[int] = []
        for ch in text:
            if ch not in self._char_to_id:
                raise ValueError(f"character {ch!r} is not in the vocabulary")
            ids.append(self._char_to_id[ch])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token ids back to text, ignoring blank tokens."""
        chars = []
        for token in ids:
            if token == self.BLANK:
                continue
            if token not in self._id_to_char:
                raise ValueError(f"token id {token} is not in the vocabulary")
            chars.append(self._id_to_char[token])
        return "".join(chars)
