#!/usr/bin/env python

# export a RooWorkspace from a ROOT file to HS3 JSON

import sys
from argparse import ArgumentParser

import ROOT


def export(path, workspace=None):
    """Return the HS3 JSON string for a RooWorkspace (default: the only one in the file)."""
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
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
    parser = ArgumentParser(description="export a RooWorkspace to HS3 JSON")
    parser.add_argument("--workspace", default=None, help="default: the only workspace in the file")
    parser.add_argument("input", help="ROOT file")
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
