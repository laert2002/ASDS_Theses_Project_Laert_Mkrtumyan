"""
kvlab - a small research library for studying memory-computation trade-offs
in long-context LLM inference, centred on the KV cache.

Sub-modules (import them explicitly, nothing heavy is imported here):
    memory_model  analytic formulas: weights, KV cache, FLOPs, roofline (pure Python)
    measure       GPU timing / peak-memory helpers
    compat        read/write the Hugging Face KV cache across library versions
    scoring       custom attention backend that also records attention statistics
    policies      KV-cache policies: eviction, quantization, mixed precision, ...
    runner        prefill/decode loop that applies a policy after every forward pass
    data          long-text loading, fixed-length prompts, passkey (needle) prompts
    quality       perplexity and passkey-retrieval evaluation under a policy
    io            config loading and result saving
"""

__version__ = "0.1.0"
