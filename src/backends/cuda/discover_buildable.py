#!/usr/bin/env python3
"""Emit a GitHub Actions matrix of CUDA backends that can be compile-checked.

A backend is compile-checkable when its manifest says more than ``planned`` and
names a ``build.script`` that exists. The workflow uses this so adding a backend
does not require editing CI: the manifest is the single source of truth.

The value is written to stdout as JSON in the shape GitHub expects:
``{"include": [{"name": ..., "dir": ..., "output": ...}, ...]}``, and to
``$GITHUB_OUTPUT`` as ``matrix=...`` when that file is set.

Usage:
    python3 src/backends/cuda/discover_buildable.py [--repo-root PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_contract  # noqa: E402  (path is set above)


def buildable(repo_root):
    """Backends that are worth compiling: not planned, and contract-clean.

    A backend the contract check already rejects is skipped, so the compile job
    never spends a runner proving what `check_contract.py` reported instantly.
    The contract job fails the build independently; this only keeps the matrix
    honest.
    """
    manifests, _issues = check_contract.run(repo_root)
    entries = []
    for manifest in manifests:
        if manifest.get("status") == "planned":
            continue
        build = manifest.get("build") or {}
        script = build.get("script")
        if not script:
            continue
        directory = manifest.get("_directory", "")
        if not os.path.isfile(os.path.join(directory, script)):
            continue
        # Re-run the manifest checks so a backend with errors is left out.
        findings = []
        check_contract.check_manifest(manifest, findings, repo_root)
        if any(finding.level == "error" for finding in findings):
            print("skipping %s: contract errors (%s)"
                  % (manifest.get("_entry", "?"),
                     "; ".join(f.message for f in findings if f.level == "error")),
                  file=sys.stderr)
            continue
        entries.append({
            "name": manifest.get("_entry", "?"),
            "dir": os.path.relpath(directory, repo_root).replace(os.sep, "/"),
            "script": script,
            "output": build.get("output", ""),
            "default_arch": str(build.get("default_arch", "")),
        })
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    entries = buildable(os.path.abspath(args.repo_root))
    matrix = {"include": entries}
    encoded = json.dumps(matrix)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write("matrix=%s\n" % encoded)

    print(encoded)
    print("%d backend(s) can be compile-checked: %s"
          % (len(entries), ", ".join(entry["name"] for entry in entries) or "none"),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
