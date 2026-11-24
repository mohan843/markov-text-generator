# Markov Text Generator

An interpretable generative-AI mini-project that learns local word-sequence patterns with an n-gram Markov chain and generates new text through weighted next-token sampling.

## Overview

This project demonstrates a foundational language-generation technique without hiding the algorithm behind a framework. A configurable context window records which tokens follow each n-gram, then samples likely continuations using a reproducible random seed.

## Problem statement

Given a small training corpus, generate plausible text that reflects its local vocabulary and phrasing while exposing the mechanics of tokenization, conditional probability, and stochastic decoding.

## Features

- Configurable first-order and higher-order n-gram contexts.
- Frequency-weighted next-token sampling.
- Optional seed phrase for controlled generation.
- Reproducible output through a random seed.
- Human-readable tokenization and detokenization.
- CLI and Python API.
- Unit tests for learning, generation, reproducibility, and validation.

## Technology stack

- Python 3.10+
- Generative AI and natural language processing
- Markov chains and n-gram language models
- Python `random`, `re`, `collections`, `argparse`, and `unittest`
- Standard library only

## Installation

```bash
git clone https://github.com/mohan843/markov-text-generator.git
cd markov-text-generator
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party dependencies are required.

## Input format

Provide a UTF-8 plain-text corpus. The included sample contains short educational and engineering sentences. For meaningful generation, use a corpus substantially larger than the included demonstration file.

## Execution

Generate text from the sample corpus:

```bash
python generator.py data/training.txt --order 2 --length 35 --seed 7
```

Start from a known context:

```bash
python generator.py data/training.txt --order 2 --length 25 --start "The model" --seed 7
```

Use the Python API:

```python
from generator import MarkovTextGenerator

model = MarkovTextGenerator(order=2, seed=7).fit(corpus)
print(model.generate(max_tokens=50, start="The model"))
```

## Sample output

```text
Training tokens: 70
Vocabulary size: 44
Generated text: student reads the explanation, asks a clear question, and improves the design. The student reads the explanation, asks a clear question, and studies the answer. Good models learn from
```

Stochastic generation may produce different text when the seed or training corpus changes.

## Testing

```bash
python -m unittest discover -s tests -v
```

## How it works

For an order-2 model, each pair of consecutive tokens is used as a state. The model counts the tokens that followed each state in the training corpus. During generation, it samples from those counts, shifts the context window, and repeats until the requested length or an unseen context is reached.

## Limitations and responsible use

This is a teaching model, not a modern large language model. It has no semantic understanding, long-range memory, factuality checking, or safety filter, and it can reproduce patterns from its training corpus. Do not use generated text as authoritative information or train on private material without permission.

## Future improvements

- Add temperature and top-k/top-p decoding.
- Support sentence-aware stopping and unknown-token handling.
- Add corpus statistics and n-gram perplexity evaluation.
- Compare Markov generation with an RNN or Transformer baseline.
- Add content filtering and provenance metadata for generated output.
