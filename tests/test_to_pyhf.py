import math

import pyhf
import pytest
from pyhf.contrib.extended_modifiers import purefunc

import to_pyhf


def gamma_constraints(name, taus):
    """Poisson constraint terms, product functions and domain entries for per-bin parameters."""
    distributions, functions, axes = [], [], []
    for i, tau in enumerate(taus):
        par = f"{name}_bin_{i}"
        distributions.append(
            {"name": f"{par}_constraint", "type": "poisson_dist", "mean": f"{par}_poisMean", "x": f"nom_{par}"}
        )
        functions.append({"name": f"{par}_poisMean", "type": "product", "factors": [par, f"{par}_tau"]})
        axes.append({"name": f"{par}_tau", "value": tau})
    return distributions, functions, axes


@pytest.fixture
def hs3():
    stat_dists, stat_funcs, stat_axes = gamma_constraints("gamma_stat_SR", [100.0, 40000.0])
    shape_dists, shape_funcs, shape_axes = gamma_constraints("gamma_shape", [25.0, 400.0])
    staterror = {
        "name": "staterror",
        "type": "staterror",
        "parameters": ["gamma_stat_SR_bin_0", "gamma_stat_SR_bin_1"],
        "constraints": [d["name"] for d in stat_dists],
    }
    samples = [
        {
            "name": "signal",
            "data": {"contents": [2.0, 4.0]},
            "modifiers": [{"name": "kappa2", "type": "custom"}],
        },
        {
            "name": "stepped",
            "data": {"contents": [10.0, 20.0]},
            "modifiers": [
                {"name": "step_SR", "type": "custom"},
                {"name": "norm_fixed", "type": "normfactor", "parameters": ["norm_fixed"]},
            ],
        },
        {
            "name": "bkg1",
            "data": {"contents": [60.0, 300.0], "errors": [6.0, 1.2]},
            "modifiers": [
                staterror,
                {"name": "Lumi", "type": "normfactor", "parameters": ["Lumi"]},
                {"name": "norm", "type": "normfactor", "parameters": ["norm"]},
                {
                    "name": "sys",
                    "type": "normsys",
                    "parameters": ["alpha_sys"],
                    "data": {"hi": 1.1, "lo": 0.9},
                },
                {
                    "name": "shape",
                    "type": "histosys",
                    "parameters": ["alpha_shape"],
                    "data": {"hi": {"contents": [66.0, 290.0]}, "lo": {"contents": [55.0, 310.0]}},
                },
            ],
        },
        {
            "name": "bkg2",
            "data": {"contents": [40.0, 100.0], "errors": [8.0, 1.6]},
            "modifiers": [staterror],
        },
        {
            "name": "bkg3",
            "data": {"contents": [5.0, 8.0]},
            "modifiers": [
                {
                    "name": "shape_stat_bkg3",
                    "type": "shapefactor",
                    "parameters": ["gamma_shape_bin_0", "gamma_shape_bin_1"],
                    "constraints": [d["name"] for d in shape_dists],
                }
            ],
        },
    ]
    return {
        "distributions": [{"name": "model_SR", "type": "histfactory_dist", "samples": samples}]
        + stat_dists
        + shape_dists,
        "functions": [
            {"name": "kappa2", "type": "generic", "expression": "pow(kappa*cos(a*acos(-1)),2)"},
            {"name": "step_SR", "type": "step", "parameters": ["sf_bin_0", "sf_bin_1"]},
            {"name": "sf_bin_0", "type": "generic", "expression": "1+kappa"},
            {"name": "sf_bin_1", "type": "generic", "expression": "kappa2+0.5"},
        ]
        + stat_funcs
        + shape_funcs,
        "data": [{"name": "obsData_SR", "type": "binned", "contents": [120.0, 430.0]}],
        "likelihoods": [{"name": "obsData_likelihood", "distributions": ["model_SR"], "data": ["obsData_SR"]}],
        "analyses": [{"name": "analysis", "parameters_of_interest": ["kappa", "a"]}],
        "domains": [
            {
                "name": "default_domain",
                "axes": [
                    {"name": "kappa", "min": 0.0, "max": 3.0},
                    {"name": "a", "min": -1.0, "max": 1.0},
                    {"name": "norm", "min": 0.0, "max": 5.0},
                    {"name": "norm_fixed", "min": 0.0, "max": 5.0},
                    {"name": "Lumi", "min": 0.0, "max": 10.0},
                ]
                + stat_axes
                + shape_axes,
            }
        ],
        "parameter_points": [
            {
                "name": "default_values",
                "parameters": [
                    {"name": "kappa", "value": 1.0},
                    {"name": "a", "value": 0.0},
                    {"name": "norm", "value": 1.2},
                    {"name": "norm_fixed", "value": 0.9, "const": True},
                    {"name": "Lumi", "value": 1.0, "const": True},
                    {"name": "gamma_stat_SR_bin_0", "value": 1.0},
                    {"name": "gamma_stat_SR_bin_1", "value": 1.0, "const": True},  # pruned
                    {"name": "alpha_sys", "value": 1.0, "const": True},
                ],
            }
        ],
    }


def build_model(ws):
    pyhf.set_backend("jax")
    config = ws["measurements"][0]["config"]
    spec = {"channels": ws["channels"], "parameters": config["parameters"], "bindings": ws["bindings"]}
    return pyhf.Model(spec, modifier_set=purefunc.enable(), poi_name=config["poi"], validate=False)


def test_structure(hs3):
    ws = to_pyhf.convert(hs3)
    assert ws["observations"] == [{"name": "SR", "data": [120.0, 430.0]}]
    assert ws["measurements"][0]["config"]["poi"] == "kappa"
    samples = {s["name"]: s for s in ws["channels"][0]["samples"]}
    assert sorted(samples) == ["bkg1", "bkg2", "bkg3", "signal", "stepped_bin0", "stepped_bin1"]
    modifiers = {name: {m["name"]: m for m in s["modifiers"]} for name, s in samples.items()}

    # expressions
    assert modifiers["signal"]["kappa2"]["type"] == "purefunc"
    bindings = {b["names"][0]: b["expression"] for b in ws["bindings"]}
    assert bindings["kappa2"] == "(kappa * cos(a * acos(-1))) ** 2"
    assert list(bindings).index("kappa2") < list(bindings).index("sf_bin_1")  # declared before use
    # per-bin expressions split the sample
    assert samples["stepped_bin0"]["data"] == [10.0, 0.0]
    assert samples["stepped_bin1"]["data"] == [0.0, 20.0]
    assert modifiers["stepped_bin1"]["sf_bin_1"]["type"] == "purefunc"

    # per-sample uncertainties, pruned bin without uncertainty
    assert modifiers["bkg1"]["staterror_SR"]["data"] == [6.0, 0.0]
    assert modifiers["bkg2"]["staterror_SR"]["data"] == [8.0, 0.0]
    # Poisson-constrained shapefactor
    assert modifiers["bkg3"]["shape_stat_bkg3"]["type"] == "shapesys"
    assert modifiers["bkg3"]["shape_stat_bkg3"]["data"] == pytest.approx([5 / 5, 8 / 20])
    # Lumi constant at 1 is dropped, systematics lose the alpha_ prefix
    assert sorted(modifiers["bkg1"]) == ["norm", "shape", "staterror_SR", "sys"]

    parameters = {p["name"]: p for p in ws["measurements"][0]["config"]["parameters"]}
    assert parameters["norm"] == {"name": "norm", "fixed": False, "inits": [1.2], "bounds": [[0.0, 5.0]]}
    assert parameters["norm_fixed"]["fixed"] and parameters["norm_fixed"]["inits"] == [0.9]
    assert parameters["kappa"]["bounds"] == [[0.0, 3.0]]
    # constant nuisance parameter, under its pyhf name
    assert parameters["sys"] == {"name": "sys", "fixed": True, "inits": [1.0]}
    assert "shape" not in parameters  # pyhf defaults apply


def test_model(hs3):
    model = build_model(to_pyhf.convert(hs3))
    # staterror width from all samples sharing it: sqrt(6^2 + 8^2) / (60 + 40), equal to 1 / sqrt(tau)
    assert model.config.param_set("staterror_SR").sigmas[0] == pytest.approx(0.1)
    fixed = dict(zip(model.config.par_names, model.config.suggested_fixed()))
    assert not fixed["staterror_SR[0]"] and fixed["staterror_SR[1]"]
    assert fixed["norm_fixed"] and fixed["sys"]

    kappa, a = 1.5, 0.25
    pars = model.config.suggested_init()
    pars[model.config.par_slice("kappa").start] = kappa
    pars[model.config.par_slice("a").start] = a
    kappa2 = (kappa * math.cos(a * math.pi)) ** 2
    expected = [
        2 * kappa2 + 10 * 0.9 * (1 + kappa) + 60 * 1.2 * 1.1 + 40 + 5,
        4 * kappa2 + 20 * 0.9 * (kappa2 + 0.5) + 300 * 1.2 * 1.1 + 100 + 8,
    ]
    assert model.expected_actualdata(pyhf.tensorlib.astensor(pars)) == pytest.approx(expected)


def test_combined_likelihood(hs3):
    # layout written by recent ROOT versions: one combined distribution and dataset
    reference = to_pyhf.convert(hs3)
    hs3["distributions"].append(
        {"name": "simPdf", "type": "combined_dist", "components": [{"dist": "model_SR", "key": ["SR"]}]}
    )
    hs3["data"].append(
        {"name": "obsData", "type": "combined", "components": [{"dataset": "obsData_SR", "key": ["SR"]}]}
    )
    hs3["likelihoods"] = [{"name": "obsData_likelihood", "distributions": ["simPdf"], "data": ["obsData"]}]
    assert to_pyhf.convert(hs3) == reference


def test_free_lumi(hs3):
    hs3["parameter_points"][0]["parameters"][4] = {"name": "Lumi", "value": 1.0}
    hs3["parameter_points"][0]["parameters"].append({"name": "nominalLumi", "value": 1.0, "const": True})
    hs3["distributions"].append(
        {"name": "lumiConstraint", "type": "gaussian_dist", "x": "Lumi", "mean": "nominalLumi", "sigma": 0.02}
    )
    hs3["distributions"][0]["samples"][2]["modifiers"][1]["constraints"] = ["lumiConstraint"]
    ws = to_pyhf.convert(hs3)
    parameters = {p["name"]: p for p in ws["measurements"][0]["config"]["parameters"]}
    assert parameters["lumi"] == {
        "name": "lumi",
        "fixed": False,
        "inits": [1.0],
        "bounds": [[0.0, 10.0]],
        "auxdata": [1.0],
        "sigmas": [0.02],
    }
    assert build_model(ws).config.param_set("lumi").sigmas == [0.02]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("a*x^2", "a * x ** 2"),
        ("pow(x, 2)+fabs(y)", "x ** 2 + Abs(y)"),
        ("min(x, y)*exp(-x)", "Min(x, y) * exp(-x)"),
    ],
)
def test_expression(expression, expected):
    assert to_pyhf.convert_expression(expression)[0] == expected


@pytest.mark.parametrize("expression", ["x > 0 ? 1 : 2", "TMath::Pi()*x", "erf(x)"])
def test_unsupported_expression(hs3, expression):
    hs3["functions"][0]["expression"] = expression
    with pytest.raises(ValueError, match="unsupported expression for kappa2"):
        to_pyhf.convert(hs3)


def test_errors(hs3):
    with pytest.raises(ValueError, match="likelihoods matching"):
        to_pyhf.convert(hs3, dataset="asimovData")
    with pytest.raises(ValueError, match="HS3 before version 0.3 is not supported"):
        to_pyhf.convert(hs3 | {"metadata": {"hs3_version": "0.2"}})
    hs3["distributions"][0]["samples"][2]["modifiers"][3]["interpolation"] = {"in": "poly1", "out": "poly1"}
    with pytest.raises(ValueError, match="unsupported interpolation for sys"):
        to_pyhf.convert(hs3)
