"""
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

CLIP tokenization utilities.

Adapted from cxr_concept/clip.py. Provides tokenize() function
used to encode concept text for CLEAR concept scoring.

Only includes the tokenization function (not model loading/download)
since model loading is handled by concept_scorer.py.

NOTE vs original clip.py: overflow behavior changed from raising
RuntimeError to truncation + eot_token, matching train.py:preprocess_text().
This is safer for long MIMIC observations (300K+ concepts).
"""

from typing import Union, List

import torch

from clear.simple_tokenizer import SimpleTokenizer as _Tokenizer

__all__ = ["tokenize"]

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """
    Tokenize text for CLIP text encoder.

    Adapted from cxr_concept/clip.py:tokenize().
    Overflow handling follows train.py:preprocess_text() (truncate)
    instead of clip.py (raise RuntimeError).

    Parameters
    ----------
    texts : Union[str, List[str]]
        Input string(s) to tokenize
    context_length : int
        Max token length (CLIP uses 77)

    Returns
    -------
    torch.LongTensor of shape [num_texts, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            # Truncate and ensure eot_token at end (matches train.py:preprocess_text)
            # Original clip.py raises RuntimeError here instead
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
