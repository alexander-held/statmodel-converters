#!/usr/bin/env python

# convert HistFactory models to pyhf JSON, from HS3 JSON (as written by RooFit) or a RooWorkspace
# adapted from https://gitlab.cern.ch/cburgard/RooFitUtils/-/blob/master/scripts/json-roofit2pyhf.py
# expressions are written as purefunc modifiers, see https://github.com/scikit-hep/pyhf/pull/2717

import ast
import json
import sys
from argparse import ArgumentParser
from math import sqrt


# functions allowed in expressions and their sympy names
FUNCTIONS = {name: name for name in "sin cos tan asin acos atan atan2 sinh cosh tanh exp log sqrt".split()}
FUNCTIONS.update({"abs": "Abs", "fabs": "Abs", "min": "Min", "max": "Max"})


class ToSympy(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
        if name == "pow" and len(node.args) == 2:
            return ast.BinOp(node.args[0], ast.Pow(), node.args[1])  # sympy does not know pow(a, b)
        if name not in FUNCTIONS:
            raise ValueError(f"unsupported function {name}")
        node.func = ast.Name(FUNCTIONS[name])
        return node


def convert_expression(expression):
    """Translate a RooFit formula to sympy syntax, returns it and the names it depends on."""
    tree = ast.parse(expression.replace("^", "**"), mode="eval")  # ^ is a power in TFormula
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - called
    return ast.unparse(ToSympy().visit(tree)), names


# HistFactory default interpolation, to use in pyhf as code4 (normsys) and code4p (histosys)
INTERPOLATION = {
    "normsys": {"in": "poly6", "out": "exp", "type": "mult"},
    "histosys": {"in": "poly6", "out": "poly1", "type": "add"},
}


def only_bin(values, i):
    return [v if j == i else 0.0 for j, v in enumerate(values)]


def split_by_bin(sample, bin_functions):
    """Split a sample into one sample per bin to apply a different purefunc modifier to each bin."""
    samples = []
    for i in range(len(sample["data"])):
        modifiers = []
        for modifier in sample["modifiers"]:
            modifier = dict(modifier)
            if modifier["type"] == "histosys":
                modifier["data"] = {k: only_bin(v, i) for k, v in modifier["data"].items()}
            elif modifier["type"] in ("staterror", "shapesys"):
                modifier["data"] = only_bin(modifier["data"], i)
            if modifier["type"] == "shapesys":
                modifier["name"] += f"_bin{i}"  # shapesys cannot be shared between samples
            modifiers.append(modifier)
        modifiers += [{"name": names[i], "type": "purefunc"} for names in bin_functions]
        name = f"{sample['name']}_bin{i}"
        samples.append({"name": name, "data": only_bin(sample["data"], i), "modifiers": modifiers})
    return samples


def hs3_version(hs3):
    version = hs3.get("metadata", {}).get("hs3_version", "0.3")
    return tuple(int(v) for v in version.split(".")[:2])


def load_hs3(path, workspace=None):
    """Read HS3 from a JSON file, going through ROOT for ROOT files and HS3 before version 0.3."""
    if path.endswith(".json"):
        with open(path) as f:
            hs3 = json.load(f)
        if hs3_version(hs3) >= (0, 3):
            return hs3
    import to_hs3  # needs ROOT

    return json.loads(to_hs3.export(path, workspace))


def convert(hs3, dataset="obsData", poi=None, fix_other_pois=False):
    """Convert a HistFactory workspace from HS3 (dict) to a pyhf workspace (dict)."""
    if hs3_version(hs3) < (0, 3):
        raise ValueError("HS3 before version 0.3 is not supported, update the file with to_hs3.py")
    bounds = {}
    constants = {}
    for domain in hs3.get("domains", []):
        for axis in domain["axes"]:
            if "min" in axis and "max" in axis:
                bounds[axis["name"]] = [axis["min"], axis["max"]]
            if "value" in axis:
                constants[axis["name"]] = axis["value"]
    defaults = {}
    for point in hs3.get("parameter_points", []):
        if point["name"] == "default_values":
            defaults = {p["name"]: p for p in point["parameters"]}
    functions = {f["name"]: f for f in hs3.get("functions", [])}
    distributions = {d["name"]: d for d in hs3["distributions"]}

    # the likelihood of the requested dataset pairs up distributions and data
    likelihoods = [lh for lh in hs3["likelihoods"] if dataset in lh["name"]]
    if len(likelihoods) != 1:
        raise ValueError(f"found {len(likelihoods)} likelihoods matching {dataset}")
    data = {d["name"]: d for d in hs3["data"]}
    data_names = {}
    for dist, dataset_name in zip(likelihoods[0]["distributions"], likelihoods[0]["data"]):
        if distributions[dist]["type"] == "combined_dist":
            # channels of a combined distribution and dataset are matched by key
            datasets = {tuple(c["key"]): c["dataset"] for c in data[dataset_name]["components"]}
            for component in distributions[dist]["components"]:
                data_names[component["dist"]] = datasets[tuple(component["key"])]
        else:
            data_names[dist] = dataset_name

    parameters = {}  # parameters to write to the measurement, by name
    bindings = {}  # expressions used by purefunc modifiers, by name

    def add_parameter(name, hs3_name=None):
        """Write initial value, bounds and whether it is fixed for a parameter."""
        default = defaults.get(hs3_name or name, {})
        parameters[name] = {"name": name, "fixed": default.get("const", False)}
        if "value" in default:
            parameters[name]["inits"] = [default["value"]]
        if (hs3_name or name) in bounds:
            parameters[name]["bounds"] = [bounds[hs3_name or name]]

    def add_systematic(mtype, parname, pdf, modifier):
        """Name of a normsys / histosys, with parameter settings where pyhf defaults do not apply."""
        interpolation = modifier.get("interpolation", pdf.get("default_interpolation"))
        if interpolation not in (None, INTERPOLATION[mtype]):
            raise ValueError(f"unsupported interpolation for {modifier['name']}: {interpolation}")
        name = parname.replace("alpha_", "")
        default = defaults.get(parname, {})
        if default.get("const", False) or default.get("value", 0.0) != 0.0:
            add_parameter(name, parname)
        return name

    def add_binding(name):
        if name in bindings:
            return
        if functions.get(name, {}).get("type") != "generic":
            raise ValueError(f"{name} is not a generic function")
        try:
            expression, names = convert_expression(functions[name]["expression"])
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"unsupported expression for {name}: {e}")
        for dependency in sorted(names):
            if dependency in functions:
                add_binding(dependency)  # bindings need to be declared before they are used
            elif dependency not in parameters:
                add_parameter(dependency)
        bindings[name] = {"names": [name], "expression": expression, "language": "sympy"}

    def rel_uncertainties(modifier):
        """Constraint type and relative uncertainty per bin for a constrained shapefactor."""
        constraints = [distributions[name] for name in modifier["constraints"]]
        types = {constraint["type"] for constraint in constraints}
        if types == {"poisson_dist"}:
            # the mean is the product of the parameter and tau = 1 / (relative uncertainty)^2
            taus = []
            for constraint in constraints:
                factors = functions[constraint["mean"]]["factors"]
                (tau,) = [f for f in factors if f not in modifier["parameters"]]
                taus.append(constants[tau])
            return "poisson", [1 / sqrt(tau) for tau in taus]
        if types == {"gaussian_dist"}:
            sigmas = [constraint["sigma"] for constraint in constraints]
            return "gaussian", [constants[s] if isinstance(s, str) else s for s in sigmas]
        raise ValueError(f"unsupported constraints for {modifier['name']}")

    def prune(modifier, uncertainties):
        """Remove uncertainties of bins with constant parameters (pruned), pyhf then fixes them."""
        if len(modifier["parameters"]) != len(uncertainties):
            raise ValueError(f"{modifier['name']} does not have one parameter per bin")
        is_const = [defaults.get(p, {}).get("const", False) for p in modifier["parameters"]]
        return [0.0 if const else u for const, u in zip(is_const, uncertainties)]

    channels = []
    observations = []
    for pdf in sorted(hs3["distributions"], key=lambda x: x["name"]):
        if pdf["type"] != "histfactory_dist":
            continue
        channelname = pdf["name"].removeprefix("model_")
        if pdf["name"] not in data_names:
            raise ValueError(f"no data for {pdf['name']} in {likelihoods[0]['name']}")
        observations.append({"name": channelname, "data": data[data_names[pdf["name"]]]["contents"]})

        samples = []
        for sample in sorted(pdf["samples"], key=lambda x: x["name"]):
            values = sample["data"]["contents"]
            modifiers = []
            bin_functions = []  # names of bindings per bin, for expressions that differ by bin
            for modifier in sorted(sample["modifiers"], key=lambda x: x["name"]):
                mtype = modifier["type"]
                parname = modifier["parameters"][0] if modifier.get("parameters") else None
                if modifier["name"] == "Lumi":
                    lumi = defaults.get(parname, {})
                    if lumi.get("const", False) and lumi.get("value", 1.0) == 1.0:
                        continue  # fixed at nominal, has no effect
                    modifiers.append({"name": "lumi", "type": "lumi", "data": None})
                    add_parameter("lumi", parname)
                    if modifier.get("constraints"):
                        # Gaussian constraint, the global observable is the term that is not the parameter
                        constraint = distributions[modifier["constraints"][0]]
                        (nominal,) = {constraint["mean"], constraint["x"]} - {parname}
                        sigma = constraint["sigma"]
                        parameters["lumi"]["auxdata"] = [defaults.get(nominal, {}).get("value", 1.0)]
                        parameters["lumi"]["sigmas"] = [constants[sigma] if isinstance(sigma, str) else sigma]
                elif mtype == "normfactor":
                    modifiers.append({"name": parname, "type": "normfactor", "data": None})
                    add_parameter(parname)
                elif mtype == "normsys":
                    name = add_systematic(mtype, parname, pdf, modifier)
                    modifiers.append({"name": name, "type": "normsys", "data": modifier["data"]})
                elif mtype == "histosys":
                    name = add_systematic(mtype, parname, pdf, modifier)
                    hi_lo = {
                        "hi_data": modifier["data"]["hi"]["contents"],
                        "lo_data": modifier["data"]["lo"]["contents"],
                    }
                    modifiers.append({"name": name, "type": "histosys", "data": hi_lo})
                elif mtype == "staterror":
                    # pyhf combines the uncertainties of all samples sharing the modifier
                    name = "staterror_" + channelname
                    uncertainties = prune(modifier, sample["data"]["errors"])
                    modifiers.append({"name": name, "type": "staterror", "data": uncertainties})
                elif mtype == "shapefactor":
                    name = modifier["name"]
                    if not modifier.get("constraints"):
                        modifiers.append({"name": name, "type": "shapefactor", "data": None})
                        continue
                    # constrained per-bin parameters of a single sample (HistFactory ShapeSys)
                    constraint, rel = rel_uncertainties(modifier)
                    uncertainties = prune(modifier, [r * v for r, v in zip(rel, values)])
                    # pyhf only has Gaussian constraints for staterror and Poisson for shapesys
                    mtype = "shapesys" if constraint == "poisson" else "staterror"
                    modifiers.append({"name": name, "type": mtype, "data": uncertainties})
                elif mtype == "custom":
                    function = functions.get(modifier["name"], {})
                    if function.get("type") == "step":
                        # one function per bin
                        if len(function["parameters"]) != len(values):
                            raise ValueError(f"{modifier['name']} does not have one entry per bin")
                        for name in function["parameters"]:
                            add_binding(name)
                        bin_functions.append(function["parameters"])
                    else:
                        add_binding(modifier["name"])
                        modifiers.append({"name": modifier["name"], "type": "purefunc"})
                else:
                    raise ValueError(f"unknown modifier type {mtype}")
            sample = {"name": sample["name"], "data": values, "modifiers": modifiers}
            samples += split_by_bin(sample, bin_functions) if bin_functions else [sample]
        channels.append({"name": channelname, "samples": samples})

    pois = []
    for analysis in hs3["analyses"]:
        pois += [poi for poi in analysis["parameters_of_interest"] if poi not in pois]
    poi = poi or (pois[0] if pois else None)
    if fix_other_pois:
        for other in pois:
            if other != poi and other in parameters:
                parameters[other]["fixed"] = True

    config = {"parameters": [parameters[name] for name in sorted(parameters)]}
    if poi:
        config["poi"] = poi
    output = {
        "channels": channels,
        "measurements": [{"name": "meas", "config": config}],
        "observations": observations,
        "version": "1.0.0",
    }
    if bindings:
        output["bindings"] = list(bindings.values())
    return output


def main():
    parser = ArgumentParser(description="convert HistFactory models from HS3 JSON or a RooWorkspace to pyhf JSON")
    parser.add_argument("--workspace", default=None, help="for ROOT files, default: the only workspace")
    parser.add_argument("--poi", default=None, help="default: first POI in the model")
    parser.add_argument("--data", default="obsData", help="name of the dataset")
    parser.add_argument("--fix-other-pois", action="store_true")
    parser.add_argument("input", help="HS3 JSON file or ROOT file")
    parser.add_argument("output", help="pyhf JSON output")
    args = parser.parse_args()
    try:
        hs3 = load_hs3(args.input, args.workspace)
        output = convert(hs3, args.data, args.poi, args.fix_other_pois)
    except ValueError as e:
        sys.exit(f"{args.input}: {e}")
    with open(args.output, "w") as f:
        json.dump(output, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
