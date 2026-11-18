#!/usr/bin/env python

# write HS3 JSON with the current ROOT version, from a RooWorkspace in a ROOT file or from an HS3 JSON
# file written by an older ROOT version

import sys
from argparse import ArgumentParser

import ROOT


def export(path, workspace=None):
    """Return the HS3 JSON string for a ROOT file (default: its only workspace) or an HS3 JSON file."""
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
    if path.endswith(".json"):
        ws = ROOT.RooWorkspace(workspace or "combined")
        if not ROOT.RooJSONFactoryWSTool(ws).importJSON(path):
            raise ValueError(f"cannot import {path}")
        return str(ROOT.RooJSONFactoryWSTool(ws).exportJSONtoString())
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise ValueError(f"cannot open {path}")
    if workspace is None:
        names = [k.GetName() for k in f.GetListOfKeys() if k.GetClassName() == "RooWorkspace"]
        if len(names) != 1:
            raise ValueError(f"found workspaces {names}, pick one with --workspace")
        workspace = names[0]
    ws = f.Get(workspace)
    if not isinstance(ws, ROOT.RooWorkspace):
        raise ValueError(f"no RooWorkspace named {workspace}")
    return str(ROOT.RooJSONFactoryWSTool(ws).exportJSONtoString())


def main():
    parser = ArgumentParser(description="write a RooWorkspace or an older HS3 JSON file as current HS3 JSON")
    parser.add_argument("--workspace", default=None, help="for ROOT files, default: the only workspace")
    parser.add_argument("input", help="ROOT file, or HS3 JSON file from an older ROOT version")
    parser.add_argument("output", help="HS3 JSON output")
    args = parser.parse_args()
    try:
        hs3 = export(args.input, args.workspace)
    except ValueError as e:
        sys.exit(f"{args.input}: {e}")
    with open(args.output, "w") as f:
        f.write(hs3)


if __name__ == "__main__":
    main()
