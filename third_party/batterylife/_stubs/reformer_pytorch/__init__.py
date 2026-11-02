"""Stub for reformer_pytorch, imported at module level by layers/SelfAttention_Family.py.

Only ReformerLayer uses LSHSelfAttention; CPMLP and CPTransformer never construct it. This stub is put on sys.path
only when the real package is not installed, so the upstream file stays unchanged.
"""


class LSHSelfAttention:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("reformer_pytorch is not installed; LSHSelfAttention is unavailable")
