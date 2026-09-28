#!/usr/bin/env python3
"""Tests for the compile-matrix discovery.

Discovery reads the manifests, so its job is to agree with the contract check:
anything the contract rejects must not reach the compile job, and anything the
manifest declares must be found without editing CI.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import discover_buildable  # noqa: E402  (path is set above)


BUILD_SH = """#!/usr/bin/env bash
set -euo pipefail
out=${{1:?usage: build.sh <output dir> [compute capability]}}
arch=${{2:-${{CUDA_COMPUTE_CAP:-89}}}}
nvcc -O3 -gencode "arch=compute_${{arch}},code=[sm_${{arch}},compute_${{arch}}]" \\
    -shared -o "$out/lib{name}_cuda.so" ./*.cu
"""


def add_backend(root, name, status="validated", executable=True, output=None, arch=89):
    directory = os.path.join(root, "src", "backends", "cuda", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "kernels.cu"), "w", encoding="utf-8") as handle:
        handle.write("// kernel\n")
    script = os.path.join(directory, "build.sh")
    with open(script, "w", encoding="utf-8") as handle:
        # The architecture is written literally, so the script really does declare
        # the target the manifest claims; `$2` still overrides it.
        handle.write(BUILD_SH.format(name=name).replace(
            "arch=${2:-${CUDA_COMPUTE_CAP:-89}}", "arch=${2:-%d}" % arch))
    os.chmod(script, 0o755 if executable else 0o644)
    manifest = {
        "name": name,
        "abi_version": 1,
        "status": status,
        "sources": ["kernels.cu"],
        "build": {
            "script": "build.sh",
            "output": output or "lib%s_cuda.so" % name,
            "default_arch": arch,
            "architectures": [arch],
        },
    }
    if status == "validated":
        manifest["numerics"] = {"tolerance": {"max_abs": 0.039}}
        manifest["reference"] = {"entrypoint": "recipe/check.py"}
        reference = os.path.join(root, "recipe", "check.py")
        os.makedirs(os.path.dirname(reference), exist_ok=True)
        if not os.path.isfile(reference):
            with open(reference, "w", encoding="utf-8") as handle:
                handle.write("import sys\nsys.exit(0)\n")
    with open(os.path.join(directory, name + ".backend.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return directory


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="omni-discover-")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def buildable(self):
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            entries = discover_buildable.buildable(self.root)
        return entries, buffer.getvalue()

    def test_a_validated_backend_is_discovered(self):
        add_backend(self.root, "qwen3_5")
        entries, _ = self.buildable()
        self.assertEqual([entry["name"] for entry in entries], ["qwen3_5"])
        self.assertEqual(entries[0]["output"], "libqwen3_5_cuda.so")
        self.assertEqual(entries[0]["default_arch"], "89")
        self.assertEqual(entries[0]["dir"], "src/backends/cuda/qwen3_5")

    def test_a_planned_backend_is_not_compiled(self):
        add_backend(self.root, "laya", status="planned")
        entries, _ = self.buildable()
        self.assertEqual(entries, [])

    def test_a_backend_with_contract_errors_is_not_compiled(self):
        # The contract job fails the build; the compile job must not also spend
        # a runner on it.
        add_backend(self.root, "laya", output="libwrong.so")
        entries, notes = self.buildable()
        self.assertEqual(entries, [])
        self.assertIn("contract errors", notes)

    def test_a_non_executable_build_script_is_not_compiled(self):
        add_backend(self.root, "laya", executable=False)
        entries, notes = self.buildable()
        self.assertEqual(entries, [])
        self.assertIn("not executable", notes)

    def test_several_backends_are_all_discovered(self):
        add_backend(self.root, "qwen3_5")
        add_backend(self.root, "laya", arch=90)
        entries, _ = self.buildable()
        self.assertEqual(sorted(entry["name"] for entry in entries), ["laya", "qwen3_5"])

    def test_the_matrix_is_valid_json_for_github(self):
        add_backend(self.root, "qwen3_5")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
            exit_code = discover_buildable.main(["--repo-root", self.root])
        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue().strip().splitlines()[0])
        self.assertEqual(len(payload["include"]), 1)
        self.assertEqual(payload["include"][0]["name"], "qwen3_5")

    def test_an_empty_repository_yields_an_empty_matrix(self):
        entries, _ = self.buildable()
        self.assertEqual(entries, [])



class FlatLayoutDiscoveryTest(unittest.TestCase):
    """A backend whose files sit directly under src/backends/cuda/.

    Laya's kernels and tools are `src/backends/cuda/kernels/` and
    `src/backends/cuda/tools/` rather than a `<name>/` subdirectory, so the
    manifest sits beside them and the compile matrix has to report the cuda
    directory itself.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="omni-flat-discover-")
        cuda = os.path.join(self.root, "src", "backends", "cuda")
        os.makedirs(os.path.join(cuda, "kernels"))
        os.makedirs(os.path.join(cuda, "tools"))
        with open(os.path.join(cuda, "kernels", "runtime.cu"), "w", encoding="utf-8") as handle:
            handle.write("// kernel\n")
        script = os.path.join(cuda, "build.sh")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env bash\n"
                         "out=${1:?usage}\n"
                         "arch=${2:-90}\n"
                         'nvcc -gencode "arch=compute_${arch}" '
                         '-shared -o "$out/liblaya_cuda.so" ./*.cu\n')
        os.chmod(script, 0o755)
        manifest = {
            "name": "laya",
            "abi_version": 1,
            "status": "experimental",
            "sources": ["kernels/runtime.cu"],
            "build": {
                "script": "build.sh",
                "output": "liblaya_cuda.so",
                "default_arch": 90,
                "architectures": [90],
                "min_capability": 90,
            },
        }
        with open(os.path.join(cuda, "laya.backend.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_flat_backend_is_compile_checkable(self):
        with contextlib.redirect_stderr(io.StringIO()):
            entries = discover_buildable.buildable(self.root)
        self.assertEqual([entry["name"] for entry in entries], ["laya"])
        self.assertEqual(entries[0]["dir"], "src/backends/cuda")
        self.assertEqual(entries[0]["output"], "liblaya_cuda.so")


if __name__ == "__main__":
    unittest.main(verbosity=2)
