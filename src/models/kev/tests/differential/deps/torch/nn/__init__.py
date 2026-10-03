class Module:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return None
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return _passthrough
def _passthrough(*a, **k): return None
class Linear(Module):
    def __init__(self, *a, **k): pass
def __getattr__(name):
    if name == "functional":
        import torch.nn.functional as m; return m
    return Module
