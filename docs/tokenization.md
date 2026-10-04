# Tokenization: English vs Armenian

**Script:** [src/tokenization_demo.py](../src/tokenization_demo.py) &nbsp;|&nbsp; **Raw output:** `results/tokenization.json`

Tokenizer: `meta-llama/Llama-3.2-1B`. Llama-3.1-8B uses exactly the same tokenizer, so these results apply to both models. It is a **byte-level BPE** tokenizer (tiktoken-style) with 128,000 regular tokens plus 256 special tokens, so `len(tokenizer)` = 128,256.

## 1. What a tokenizer does

A model cannot read text directly. It reads integer **token IDs**. The tokenizer:

1. converts the text to UTF-8 **bytes**,
2. repeatedly merges frequent byte sequences into larger units (**BPE merges**, learned from training data),
3. maps each resulting unit to an ID in the vocabulary.

"Byte-level" means every possible string can be encoded, because in the worst case each byte becomes its own token. Nothing is ever "unknown", but rare text becomes **long**.

How to read the script output:

| Field | Meaning |
|---|---|
| `ids` | Integers the model actually sees. |
| `pieces` | Raw vocabulary entries. `Ġ` marks a leading space, e.g. `Ġcapital` = " capital". Non-ASCII bytes are shown as Latin-1-looking symbols (e.g. `Õ`, `Ô`), because each raw byte is mapped to a printable character. |
| `decoded` | The text each single token represents. `�` means the token holds only part of a multi-byte UTF-8 character. The character is complete only when combined with its neighbour token. |

Special tokens: when `add_special_tokens=True` (the default), the tokenizer prepends `<|begin_of_text|>` (ID 128000). We turn this off when counting.

## 2. Sentences (parallel pairs)

| # | English | Armenian |
|---|---|---|
| 1 | Yerevan is the capital of Armenia. | Երևանը Հայաստանի մայրաքաղաքն է։ |
| 2 | I love reading books. | Ես սիրում եմ կարդալ գրքեր։ |
| 3 | The weather is very nice today. | Այսօր եղանակը շատ լավ է։ |
| 4 | We study at the university. | Մենք սովորում ենք համալսարանում։ |
| 5 | Artificial intelligence is developing rapidly. | Արհեստական բանականությունը արագ զարգանում է։ |

## 3. Results

> TODO: run `python src/tokenization_demo.py` and paste the SUMMARY table here.

| Language | Words | Tokens | Tokens/word | Chars/token | Bytes/token |
|---|---|---|---|---|---|
| english | … | … | … | … | … |
| armenian | … | … | … | … | … |

Armenian / English token ratio for the same content: **… ×**

Example token breakdown (copy one interesting Armenian sentence from the output):

```
TODO
```

## 4. Answers

**How many tokens does an average English word take?**
→ *fill in* (expected ≈ 1.1–1.4). Most common English words are a single token. Only rarer or long words such as "Artificial" or "rapidly" get split.

**How many tokens does an average Armenian word take?**
→ *fill in* (expected to be several times larger than for English).

**Why the difference?**

1. **Training-data imbalance.** BPE merges are learned from the tokenizer's training corpus, which is dominated by English and a few other high-resource languages. Frequent English words were merged into single tokens. Armenian byte sequences were rarely seen, so few merges exist for them and Armenian text falls back to short pieces, sometimes single bytes.
2. **UTF-8 cost.** Every Armenian letter (Unicode block U+0530–U+058F) takes **2 bytes** in UTF-8, while an English letter takes 1. A byte-level tokenizer starts from bytes, so with no merges an Armenian word is at least twice as long as an English word with the same number of letters. The `�` pieces in the output are tokens that cut a 2-byte letter in half.
3. **Morphology.** Armenian is highly inflected and agglutinative. One word often holds what English says in several words, e.g. *համալսարանում* = "at the university" (stem + locative suffix), *մայրաքաղաքն* = "the capital" (mother + city + definite article). Long, varied word forms are less likely to exist as whole tokens.

## 5. Why this matters for the thesis

The context length, compute, and KV-cache memory of an LLM all scale with the **number of tokens**, not the number of words. If Armenian needs *k*× more tokens for the same content, then for Armenian text:

- the context window holds *k*× less content,
- the KV cache for the same document is *k*× larger (see [kv_cache_structure.md](kv_cache_structure.md)),
- attention compute, which is quadratic in tokens, is up to *k*²× larger in the prefill phase (see [shapes.md](shapes.md)).

So low-resource languages hit long-context memory limits much earlier. This is a direct motivation for studying memory–computation trade-offs.
