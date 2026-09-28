#!/usr/bin/env python3
"""Tests for the Tier-2 GPU parity driver.

There is no GPU here, so the CUDA probe is stubbed. What is under test is the
driver's *decisions*: when to refuse, when to skip, and what it reports. The one
thing that must never happen is a silent pass on a machine that cannot run CUDA.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_gpu_parity  # noqa: E402  (path is set above)


def build_repo(root, status="validated", entrypoint="recipe/cua_s1/check.py",
               script="import sys\nsys.exit(0)\n"):
    """A minimal repository with one backend manifest and a reference script."""
    backend = os.path.join(root, "src", "backends", "cuda", "qwen3_5")
    os.makedirs(backend)
    for name in ("kernels.cu", "ops.h"):
        with open(os.path.join(backend, name), "w", encoding="utf-8") as handle:
            handle.write("// fixture\n")
    with open(os.path.join(backend, "build.sh"), "w", encoding="utf-8") as handle:
        handle.write('#!/usr/bin/env bash\n'
                     'arch=${1:-89}\n'
                     'nvcc -gencode "arch=compute_${arch},code=sm_${arch}" '
                     '-shared -o libqwen3_5_cuda.so ./*.cu\n')
    os.chmod(os.path.join(backend, "build.sh"), 0o755)
    manifest = {
        "name": "qwen3_5",
        "abi_version": 1,
        "status": status,
        "sources": ["ops.h", "kernels.cu"],
        "build": {"script": "build.sh", "output": "libqwen3_5_cuda.so",
                  "default_arch": 89, "architectures": [89]},
        "numerics": {"tolerance": {"max_abs": 0.039}},
        "reference": {"entrypoint": entrypoint},
    }
    with open(os.path.join(backend, "qwen3_5.backend.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    if entrypoint:
        path = os.path.join(root, entrypoint)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(script)
    return root


class DriverTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="omni-gpu-")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def run_main(self, argv):
        buffer = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(errors):
            code = run_gpu_parity.main(argv)
        return code, buffer.getvalue(), errors.getvalue()

    def test_lists_validated_backends_without_needing_a_gpu(self):
        build_repo(self.root)
        code, out, _ = self.run_main(["--repo-root", self.root, "--list"])
        self.assertEqual(code, 0)
        self.assertIn("qwen3_5 -> recipe/cua_s1/check.py", out)

    def test_no_validated_backend_is_a_clean_no_op(self):
        build_repo(self.root, status="experimental")
        code, out, _ = self.run_main(["--repo-root", self.root])
        self.assertEqual(code, 0)
        self.assertIn("nothing to run on a GPU", out)

    def test_a_missing_gpu_refuses_instead_of_passing(self):
        build_repo(self.root)
        with mock.patch.object(run_gpu_parity, "has_usable_gpu",
                               return_value=(False, "nvidia-smi is not on PATH")):
            code, _, errors = self.run_main(["--repo-root", self.root])
        self.assertEqual(code, 2, "a GPU-less machine must not report success")
        self.assertIn("needs a GPU", errors)

    def test_no_devices_reported_refuses(self):
        build_repo(self.root)
        with mock.patch.object(run_gpu_parity, "has_usable_gpu",
                               return_value=(False, "nvidia-smi reports no devices")):
            code, _, errors = self.run_main(["--repo-root", self.root])
        self.assertEqual(code, 2)
        self.assertIn("no devices", errors)

    def test_a_passing_reference_reports_pass_with_provenance(self):
        build_repo(self.root)
        provenance = {"driver": "595.91.07", "gpu": ["NVIDIA H100 PCIe"],
                      "cuda_toolkit": "Cuda compilation tools, release 13.2",
                      "compute_capability": "9.0"}
        with mock.patch.object(run_gpu_parity, "has_usable_gpu",
                               return_value=(True, "GPU 0: NVIDIA H100 PCIe")), \
                mock.patch.object(run_gpu_parity, "gpu_provenance", return_value=provenance):
            report_path = os.path.join(self.root, "parity.json")
            code, out, _ = self.run_main(["--repo-root", self.root, "--json", report_path])
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
        self.assertEqual(code, 0)
        self.assertIn("PASS qwen3_5", out)
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["provenance"]["driver"], "595.91.07")
        self.assertEqual(report["results"][0]["tolerance"], {"max_abs": 0.039})

    def test_a_failing_reference_fails_the_run(self):
        build_repo(self.root, script="import sys\nprint('mismatch: 0.4 > 0.039')\nsys.exit(1)\n")
        with mock.patch.object(run_gpu_parity, "has_usable_gpu",
                               return_value=(True, "GPU 0")), \
                mock.patch.object(run_gpu_parity, "gpu_provenance",
                                  return_value={"gpu": ["fake"], "driver": "", "cuda_toolkit": "",
                                                "compute_capability": ""}):
            code, out, _ = self.run_main(["--repo-root", self.root])
        self.assertEqual(code, 1)
        self.assertIn("FAIL qwen3_5", out)

    def test_only_filters_to_one_backend(self):
        build_repo(self.root)
        code, out, _ = self.run_main(["--repo-root", self.root, "--list", "--only", "laya"])
        self.assertEqual(code, 0)
        self.assertIn("no validated backends declared", out)

    def test_a_declared_but_absent_entrypoint_is_an_error(self):
        build_repo(self.root)
        # Declare a reference that is not on disk: parity must stop rather than
        # report a pass it never ran.
        os.remove(os.path.join(self.root, "recipe", "cua_s1", "check.py"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            run_gpu_parity.main(["--repo-root", self.root])
        self.assertIn("does not exist", str(caught.exception))

    def test_a_contract_error_stops_parity_before_any_gpu_work(self):
        build_repo(self.root)
        manifest_path = os.path.join(self.root, "src", "backends", "cuda", "qwen3_5",
                                     "qwen3_5.backend.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["build"]["output"] = "libwrong.so"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            run_gpu_parity.main(["--repo-root", self.root])
        self.assertIn("contract check failed", str(caught.exception))


class GpuProbeTest(unittest.TestCase):
    """The probe must not claim a GPU it cannot see."""

    def test_probe_reports_failure_without_nvidia_smi(self):
        with mock.patch.object(run_gpu_parity.shutil, "which", return_value=None):
            usable, detail = run_gpu_parity.has_usable_gpu()
        self.assertFalse(usable)
        self.assertIn("nvidia-smi", detail)

    def test_probe_rejects_an_empty_device_list(self):
        with mock.patch.object(run_gpu_parity.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(run_gpu_parity.subprocess, "run") as runner:
            runner.return_value = mock.Mock(returncode=0, stdout="\n", stderr="")
            usable, detail = run_gpu_parity.has_usable_gpu()
        self.assertFalse(usable)
        self.assertIn("no devices", detail)

    def test_probe_accepts_a_listed_device(self):
        with mock.patch.object(run_gpu_parity.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(run_gpu_parity.subprocess, "run") as runner:
            runner.return_value = mock.Mock(
                returncode=0, stdout="GPU 0: NVIDIA H100 PCIe (UUID: GPU-abc)\n", stderr="")
            usable, detail = run_gpu_parity.has_usable_gpu()
        self.assertTrue(usable)
        self.assertIn("H100", detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
