#!/usr/bin/env python

# convert a HistFactory RooWorkspace from a ROOT file to pyhf JSON, via HS3

import json
import sys
from argparse import ArgumentParser

import HS3_to_pyhf
import RooWorkspace_to_HS3


def main():
    parser = ArgumentParser(description="convert a HistFactory RooWorkspace to pyhf JSON")
    parser.add_argument("--workspace", default=None, help="default: the only workspace in the file")
    parser.add_argument("--poi", default=None, help="default: first POI in the workspace")
    parser.add_argument("--data", default="obsData", help="name of the dataset")
    parser.add_argument("--fix-other-pois", action="store_true")
    parser.add_argument("input", help="ROOT file")
    parser.add_argument("output", help="pyhf JSON output")
    args = parser.parse_args()
    try:
        hs3 = json.loads(RooWorkspace_to_HS3.export(args.input, args.workspace))
        output = HS3_to_pyhf.convert(hs3, args.data, args.poi, args.fix_other_pois)
    except ValueError as e:
        sys.exit(f"{args.input}: {e}")
    with open(args.output, "w") as f:
        json.dump(output, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
