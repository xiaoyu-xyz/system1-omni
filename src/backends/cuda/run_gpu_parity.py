#!/usr/bin/env python3
"""Tier-3 GPU parity driver. Runs on a self-hosted runner that has the GPU.

For every backend whose manifest says ``status == "validated"``, this runs the
``reference.entrypoint`` the manifest declares and reports the result. It is the
Tier-3 half of ``contract.md``: the CPU-only contract check cannot compile CUDA
or prove numerics, so something with a GPU has to.

The script refuses to report success on a machine with no usable GPU. A parity
run that silently did nothing is worse than no run, which is the failure mode
``#14``'s validation report already records for the CPU workflow.

Usage:
    python3 src/backends/cuda/run_gpu_parity.py [--repo-root PATH]
                                                [--only NAME] [--json PATH]
                                                [--list]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_contract  # noqa: E402  (path is set above)


def gpu_provenance():
    """Name, driver and CUDA version of the machine this ran on.

    Reported with every result because a tolerance is only meaningful next to the
    hardware and toolkit it was measured on.
    """
    info = {"driver": "", "gpu": [], "cuda_toolkit": "", "compute_capability": ""}
    if shutil.which("nvidia-smi"):
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True)
        if query.returncode == 0:
            rows = [row.strip() for row in query.stdout.strip().splitlines() if row.strip()]
            info["gpu"] = [row.split(",")[0].strip() for row in rows]
            if rows:
                parts = [part.strip() for part in rows[0].split(",")]
                info["driver"] = parts[1] if len(parts) > 1 else ""
                info["compute_capability"] = parts[2] if len(parts) > 2 else ""
    nvcc = shutil.which("nvcc") or os.path.join(
        os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
    if os.path.isfile(nvcc):
        version = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
        if version.returncode == 0:
            for line in version.stdout.splitlines():
                if "release" in line:
                    info["cuda_toolkit"] = line.strip()
                    break
    return info


def has_usable_gpu():
    """True when the machine can actually execute CUDA work."""
    if not shutil.which("nvidia-smi"):
        return False, "nvidia-smi is not on PATH"
    probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if probe.returncode != 0:
        return False, "nvidia-smi -L failed: %s" % (probe.stderr.strip() or probe.returncode)
    if not probe.stdout.strip():
        return False, "nvidia-smi reports no devices"
    return True, probe.stdout.strip().splitlines()[0]


def validated_backends(repo_root, only=None):
    manifests, issues = check_contract.run(repo_root)
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        for issue in errors:
            print(issue, file=sys.stderr)
        raise SystemExit("contract check failed; fix the manifest before running parity")

    selected = []
    for manifest in manifests:
        entry = manifest.get("_entry", "?")
        if only and entry != only:
            continue
        if manifest.get("status") != "validated":
            print("skipping %s: status=%s (no parity claim)" % (entry, manifest.get("status")))
            continue
        reference = manifest.get("reference") or {}
        entrypoint = reference.get("entrypoint")
        if not entrypoint:
            print("skipping %s: validated but no reference.entrypoint" % entry)
            continue
        path = os.path.join(repo_root, entrypoint)
        if not os.path.isfile(path):
            raise SystemExit(
                "%s declares reference.entrypoint %r, which does not exist" % (entry, entrypoint))
        selected.append((entry, path, manifest))
    return selected


def run_backend(entry, path, manifest, repo_root):
    """Run one backend's reference entrypoint and capture its result."""
    numerics = manifest.get("numerics") or {}
    argv = [sys.executable, path]
    environment = dict(os.environ)
    environment["OMNI_BACKEND_NAME"] = entry
    environment["OMNI_REPO_ROOT"] = repo_root
    started = time.time()
    result = subprocess.run(argv, cwd=os.path.dirname(path) or repo_root,
                            capture_output=True, text=True, env=environment)
    elapsed = time.time() - started
    return {
        "backend": entry,
        "entrypoint": os.path.relpath(path, repo_root),
        "exit_code": result.returncode,
        "seconds": round(elapsed, 3),
        "tolerance": numerics.get("tolerance"),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--only", help="run one backend by name")
    parser.add_argument("--json", help="write the report to this path")
    parser.add_argument("--list", action="store_true",
                        help="list validated backends and exit without checking the GPU")
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(args.repo_root)
    selected = validated_backends(repo_root, args.only)

    if args.list:
        for entry, path, _ in selected:
            print("%s -> %s" % (entry, os.path.relpath(path, repo_root)))
        if not selected:
            print("no validated backends declared")
        return 0

    if not selected:
        print("no validated backends declared; nothing to run on a GPU")
        print("This is expected until a backend manifest sets status=validated.")
        return 0

    usable, detail = has_usable_gpu()
    if not usable:
        print("Tier-3 parity needs a GPU: %s" % detail, file=sys.stderr)
        return 2

    provenance = gpu_provenance()
    print("GPU: %s" % ", ".join(provenance["gpu"]))
    print("driver: %s  compute capability: %s" % (provenance["driver"],
                                                  provenance["compute_capability"]))
    print("toolkit: %s" % provenance["cuda_toolkit"])
    print()

    results = []
    for entry, path, manifest in selected:
        print("== %s ==" % entry)
        result = run_backend(entry, path, manifest, repo_root)
        results.append(result)
        status = "PASS" if result["exit_code"] == 0 else "FAIL"
        print("%s %s in %.1fs (tolerance %s)"
              % (status, entry, result["seconds"], result["tolerance"]))
        if result["stdout_tail"]:
            print(result["stdout_tail"])
        if result["exit_code"] != 0 and result["stderr_tail"]:
            print(result["stderr_tail"], file=sys.stderr)

    report = {
        "provenance": provenance,
        "results": results,
        "passed": sum(1 for item in results if item["exit_code"] == 0),
        "failed": sum(1 for item in results if item["exit_code"] != 0),
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("\nwrote %s" % args.json)

    print("\n%d passed, %d failed" % (report["passed"], report["failed"]))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
