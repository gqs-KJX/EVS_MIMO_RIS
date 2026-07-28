#!/usr/bin/env python
"""Compare two figure directories on content, ignoring PDF creation timestamps."""
import pathlib, re, sys

pat = re.compile(rb"/(CreationDate|ModDate)\s*\([^)]*\)")
a_dir, b_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
same = diff = missing = 0
for f in sorted(a_dir.glob("*.pdf")):
    other = b_dir / f.name
    if not other.exists():
        missing += 1
        print(f"MISSING in {b_dir}: {f.name}")
        continue
    if pat.sub(b"", f.read_bytes()) == pat.sub(b"", other.read_bytes()):
        same += 1
    else:
        diff += 1
        print(f"DIFFERS: {f.name}")
print(f"identical={same} differs={diff} missing={missing}")
sys.exit(1 if (diff or missing) else 0)
