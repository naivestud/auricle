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

    @property
    def blank_id(self) -> int:
        """Token id of the CTC blank symbol."""
        return self.BLANK

    def can_encode(self, text: str) -> bool:
        """True if every character of ``text`` is in the vocabulary."""
        return all(ch in self._char_to_id for ch in text)

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids.

        Raises ``ValueError`` naming the first character outside the
        vocabulary and its position.
        """
        ids: list[int] = []
        for i, ch in enumerate(text):
            if ch not in self._char_to_id:
                raise ValueError(f"character {ch!r} at position {i} is not in the vocabulary")
            ids.append(self._char_to_id[ch])
        return ids

    def encode_lenient(self, text: str) -> list[int]:
        """Encode ``text``, silently dropping characters outside the vocabulary.

        Useful for preparing imperfect reference text for loss computation or
        for sanitising arbitrary input before decoding round-trips.
        """
        return [self._char_to_id[ch] for ch in text if ch in self._char_to_id]

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
