#!/usr/bin/env python3
"""Fetch upstream's ``kev/model.py`` at the pinned revision for the differential test.

The file is written next to this script as ``upstream_model.py`` and is
deliberately not committed: it is someone else's source, and the comparison is
only meaningful against a stated revision.

    python3 tests/differential/fetch_upstream.py [--revision REV]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

REPO = "jaredpalmer/kev"
PATH = "kev/model.py"
# The revision the port in kev_prompt.py was written against.
REVISION = "main"
DESTINATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream_model.py")

MIRRORS = (
    "https://raw.githubusercontent.com/{repo}/{rev}/{path}",
    "https://cdn.jsdelivr.net/gh/{repo}@{rev}/{path}",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision", default=REVISION,
                        help="commit, tag or branch of %s (default: %s)" % (REPO, REVISION))
    args = parser.parse_args(argv)

    errors = []
    for template in MIRRORS:
        url = template.format(repo=REPO, rev=args.revision, path=PATH)
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                body = response.read()
        except Exception as error:  # noqa: BLE001 - report and try the next mirror
            errors.append("%s: %s" % (url, error))
            continue
        if len(body) < 1000:
            errors.append("%s: implausibly small (%d bytes)" % (url, len(body)))
            continue
        with open(DESTINATION, "wb") as handle:
            handle.write(body)
        print("wrote %s (%d bytes, sha256 %s)"
              % (DESTINATION, len(body), hashlib.sha256(body).hexdigest()[:16]))
        print("revision: %s@%s" % (REPO, args.revision))
        return 0

    for error in errors:
        print(error, file=sys.stderr)
    print("could not fetch %s/%s; the differential test will be skipped" % (REPO, PATH),
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
