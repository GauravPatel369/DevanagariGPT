"""Phase-2: decoder-only Transformer language models, built from primitives.

Shared implementation, per-language artifacts.  The project forbids sharing
data, tokenizers, vocabularies or weights between Model H and Model L -- it does
not require duplicating the code, and duplicating it would actively harm the
Phase-3 comparison: two copies that drift apart turn an architectural difference
into a confound.  One implementation, two configs, two checkpoint directories.
"""
__version__ = "2.0.0"
