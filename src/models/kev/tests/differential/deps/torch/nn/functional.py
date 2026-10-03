def softmax(*a, **k): return None
def __getattr__(name):
    def _f(*a, **k): return None
    return _f
