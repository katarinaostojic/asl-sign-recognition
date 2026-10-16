"""Turn a stream of per-frame predictions into text (letters, SPACE, DELETE).

Static fingerspelling has no natural "letter boundary", so we use a simple rule:
a letter is committed when it wins a majority vote over the last `window` frames
with at least `min_votes` votes. After a commit the builder is "disarmed" and is only
re-armed when the hand is removed (NOTHING wins the vote) or a different sign
becomes stable, so holding a sign does not repeat it. For a double letter ("LL")
lower the hand in between.
"""
from __future__ import annotations

from collections import Counter, deque


class WordBuilder:
    def __init__(
        self,
        window: int = 9,
        min_votes: int = 6,
        min_confidence: float = 0.6,
        nothing: str = "nothing",
        space: str = "space",
        delete: str = "del",
    ):
        self.window = window
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self.nothing, self.space, self.delete = nothing, space, delete
        self.reset()

    def reset(self) -> None:
        self.text = ""
        self._votes: deque = deque(maxlen=self.window)
        self._armed = True
        self._last_committed: str | None = None

    def update(self, label: str, confidence: float) -> dict:
        """Feed one frame. Returns the current text, candidate and vote progress."""
        self._votes.append(label if confidence >= self.min_confidence else None)
        counts = Counter(v for v in self._votes if v is not None)
        candidate, votes = (counts.most_common(1)[0] if counts else (None, 0))
        committed = None

        if candidate == self.nothing and votes >= self.min_votes:
            self._armed = True
            self._last_committed = None
        elif candidate is not None and votes >= self.min_votes:
            if candidate != self._last_committed:
                self._armed = True
            if self._armed:
                committed = candidate
                self._apply(candidate)
                self._armed = False
                self._last_committed = candidate
                self._votes.clear()

        return {
            "text": self.text,
            "candidate": candidate,
            "progress": min(votes / self.min_votes, 1.0),
            "committed": committed,
        }

    def _apply(self, label: str) -> None:
        if label == self.space:
            self.text += " "
        elif label == self.delete:
            self.text = self.text[:-1]
        elif label != self.nothing:
            self.text += label
