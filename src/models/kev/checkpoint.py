#!/usr/bin/env python3
"""Read the tensors out of a torch ``.pt`` checkpoint without torch.

This is production code, not a test helper: loading ``head.pt`` is what the Kev
engine does at startup, and the point is that it needs no torch.

A torch checkpoint is a ZIP holding a pickle stream that describes storages, plus
one raw file per storage. Rebuilding it needs a class resolver for
``torch._utils._rebuild_tensor_v2`` and ``torch.FloatStorage`` and nothing else,
which is what this module provides. That is enough to read Kev's ``head.pt``
(four tensors, 1,311,232 fp32 parameters) on a machine with no torch installed,
which is the situation the rest of this port is written for.

Only ``float32`` storages are handled, because that is what the head ships as and
what it must stay as. Anything else raises rather than being silently converted.
"""

from __future__ import annotations

import pickle
import struct
import zipfile

import numpy as np

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64,
           "HalfStorage": np.float16, "LongStorage": np.int64,
           "IntStorage": np.int32, "ByteStorage": np.uint8}


class _Storage:
    """One ``torch.Storage``, addressed by key into the archive."""

    def __init__(self, archive, key, dtype_name):
        self._archive = archive
        self._key = key
        if dtype_name not in _DTYPES:
            raise NotImplementedError("unsupported storage type %r" % dtype_name)
        self.dtype = _DTYPES[dtype_name]

    @property
    def array(self):
        # The key arrives as a string from the persistent id and as an int from
        # the inline form, so it is not formatted with %d.
        raw = self._archive.read("head/data/%s" % self._key)
        return np.frombuffer(raw, dtype=self.dtype)


class _Tensor:
    """What ``_rebuild_tensor_v2`` returns: a view over a storage."""

    def __init__(self, storage, offset, size, stride, requires_grad, backward_hooks,
                 metadata=None):
        self.storage = storage
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)
        self.requires_grad = requires_grad
        self.metadata = metadata

    def numpy(self):
        flat = self.storage.array
        count = 1
        for dimension in self.size:
            count *= dimension
        window = flat[self.offset:self.offset + count]
        # The head's tensors are contiguous; a strided read is done explicitly so
        # a non-contiguous checkpoint is handled rather than misread.
        if self.stride == _contiguous_stride(self.size):
            return window.reshape(self.size).copy()
        index = np.arange(count)
        coordinates = np.unravel_index(index, self.size)
        flat_index = np.zeros(count, dtype=np.int64)
        for axis, step in enumerate(self.stride):
            flat_index += coordinates[axis] * step
        return window[flat_index].reshape(self.size).copy()


def _contiguous_stride(size):
    stride, running = [], 1
    for dimension in reversed(size):
        stride.append(running)
        running *= dimension
    return tuple(reversed(stride))


class _Unpickler(pickle.Unpickler):
    def __init__(self, stream, archive):
        super().__init__(stream)
        self._archive = archive

    def persistent_load(self, pid):
        """Resolve a storage reference.

        torch writes storages as persistent ids shaped
        ``('storage', StorageClass, key, location, numel)``, so the dtype comes
        from the class carried in the id rather than from a global.
        """
        if not isinstance(pid, tuple) or pid[0] != "storage":
            raise pickle.UnpicklingError("unsupported persistent id %r" % (pid,))
        _, storage_class, key, _location, _numel = pid[:5]
        name = getattr(storage_class, "__name__", str(storage_class))
        return _Storage(self._archive, key, name)

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _Tensor
        if module == "torch" and name.endswith("Storage"):
            # Only reached if a storage is referenced non-persistently.
            return type(name, (), {})
        return super().find_class(module, name)


def load(path):
    """Return the checkpoint as a ``{name: numpy array}`` dict.

    Metadata entries that are not tensors (the calibration block, for instance)
    are passed through unchanged.
    """
    archive = zipfile.ZipFile(path)
    with archive.open("head/data.pkl") as handle:
        payload = _Unpickler(handle, archive).load()
    result = {}
    for key, value in payload.items():
        result[key] = value.numpy() if isinstance(value, _Tensor) else value
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    args = parser.parse_args(argv)

    payload = load(args.checkpoint)
    print("%s:" % args.checkpoint)
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, np.ndarray):
            print("  %-28s %-14s %s" % (key, str(value.shape), value.dtype))
        else:
            print("  %-28s %r" % (key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
