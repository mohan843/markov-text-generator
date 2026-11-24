"""Interpretable n-gram Markov text generation using the Python standard library."""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.!?,;:]")
PUNCTUATION = {".", "!", "?", ",", ";", ":"}


def tokenize(text: str) -> list[str]:
    """Convert text into words and punctuation tokens."""
    return TOKEN_PATTERN.findall(text)


def detokenize(tokens: Sequence[str]) -> str:
    """Turn tokens back into readable text."""
    output = ""
    for token in tokens:
        if token in PUNCTUATION:
            output = output.rstrip() + token
        else:
            output += (" " if output else "") + token
    return output


class MarkovTextGenerator:
    """Generate text by sampling the next token from n-gram transitions."""

    def __init__(self, order: int = 2, seed: int | None = 7) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.seed = seed
        self._rng = random.Random(seed)

    def fit(self, text: str) -> "MarkovTextGenerator":
        tokens = tokenize(text)
        if len(tokens) <= self.order:
            raise ValueError("text must contain more tokens than the selected order")
        self.tokens_ = tokens
        self.transitions_: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for index in range(len(tokens) - self.order):
            state = tuple(tokens[index : index + self.order])
            self.transitions_[state][tokens[index + self.order]] += 1
        self.start_states_ = list(self.transitions_)
        return self

    def _weighted_choice(self, counts: Counter[str]) -> str:
        choices, weights = zip(*counts.items())
        return self._rng.choices(choices, weights=weights, k=1)[0]

    def generate(self, max_tokens: int = 40, start: str | None = None) -> str:
        """Generate at most max_tokens from a fitted model."""
        if not hasattr(self, "transitions_"):
            raise ValueError("fit must be called before generate")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if start:
            prefix = tokenize(start)
            if len(prefix) < self.order:
                raise ValueError("start must contain at least order tokens")
            state = tuple(prefix[-self.order :])
            if state not in self.transitions_:
                raise ValueError("start context was not found in the training text")
            output = prefix[:]
        else:
            state = self._rng.choice(self.start_states_)
            output = list(state)
        while len(output) < max_tokens:
            next_token = self._weighted_choice(self.transitions_[state])
            output.append(next_token)
            state = tuple(output[-self.order :])
            if state not in self.transitions_:
                break
        return detokenize(output[:max_tokens])

    def vocabulary(self) -> set[str]:
        if not hasattr(self, "tokens_"):
            raise ValueError("fit must be called before vocabulary")
        return set(self.tokens_)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 training text file")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--start", help="optional seed phrase from the training text")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = args.input.read_text(encoding="utf-8")
    model = MarkovTextGenerator(order=args.order, seed=args.seed).fit(text)
    print(f"Training tokens: {len(model.tokens_)}")
    print(f"Vocabulary size: {len(model.vocabulary())}")
    print(f"Generated text: {model.generate(args.length, args.start)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
