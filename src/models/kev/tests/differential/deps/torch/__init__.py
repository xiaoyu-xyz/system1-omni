"""Torch stand-in sufficient to import kev.model.

Not a tensor library: encode(), user_tokens() and is_hybrid() are the functions
under test and they touch no tensors. Everything else exists so the module body
can execute.
"""
class _Any:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return self
    def __getattr__(self, n): return _Any()
    def __iter__(self): return iter(())
    def __bool__(self): return True

class dtype: pass
float32 = dtype(); long = dtype(); bool = dtype()

class _Cuda:
    @staticmethod
    def current_device(): return 0
cuda = _Cuda()

def _decorate(fn):
    def wrapper(*a, **k):
        return fn(*a, **k)
    return wrapper

def no_grad(*a, **k):
    """Usable both as @torch.no_grad() and as a context manager."""
    if len(a) == 1 and callable(a[0]) and not k:
        return _decorate(a[0])
    class _Ctx:
        def __enter__(self): return None
        def __exit__(self, *e): return False
        def __call__(self, fn): return _decorate(fn)
    return _Ctx()

def __getattr__(name):
    if name == "nn":
        import torch.nn as m; return m
    if name == "nn.functional":
        import torch.nn.functional as m; return m
    return _Any()
