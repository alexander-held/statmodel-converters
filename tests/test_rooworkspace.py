import json

import pyhf
import pytest
from pyhf.contrib.extended_modifiers import purefunc

import HS3_to_pyhf

ROOT = pytest.importorskip("ROOT")
import RooWorkspace_to_HS3  # noqa: E402

HISTOGRAMS = {
    "data": ([135.0, 262.0], [0.0, 0.0]),
    "signal": ([10.0, 15.0], [0.0, 0.0]),
    "background": ([100.0, 200.0], [10.0, 2.0]),  # 10% and 1% relative uncertainty
    "other": ([20.0, 40.0], [0.0, 0.0]),
    "other_rel_uncertainty": ([0.2, 0.1], [0.0, 0.0]),
    "free": ([5.0, 7.0], [0.0, 0.0]),
}


def make_workspace(tmp_path, constant):
    """Write a small HistFactory workspace, with the given parameters held constant."""
    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
    inputs = str(tmp_path / "inputs.root")
    f = ROOT.TFile.Open(inputs, "RECREATE")
    for name, (contents, errors) in HISTOGRAMS.items():
        h = ROOT.TH1F(name, name, 2, 0.0, 2.0)
        for i, (content, error) in enumerate(zip(contents, errors)):
            h.SetBinContent(i + 1, content)
            h.SetBinError(i + 1, error)
        h.Write()
    f.Close()

    hf = ROOT.RooStats.HistFactory
    measurement = hf.Measurement("meas")
    measurement.SetOutputFilePrefix(str(tmp_path / "out"))
    measurement.SetPOI("kappa")
    measurement.AddPreprocessFunction("kappa2", "pow(kappa,2)", "kappa[1,0,3]")
    measurement.SetLumi(1.0)
    measurement.SetLumiRelErr(0.02)
    for name in constant:
        measurement.AddConstantParam(name)
    channel = hf.Channel("SR")
    channel.SetData("data", inputs)
    channel.SetStatErrorConfig(0.05, "Poisson")  # prunes the second bin
    signal = hf.Sample("signal", "signal", inputs)
    signal.AddNormFactor("kappa2", 1.0, 0.0, 10.0)  # replaced by the expression
    background = hf.Sample("background", "background", inputs)
    background.ActivateStatError()
    background.AddOverallSys("bkg_norm", 0.9, 1.1)
    other = hf.Sample("other", "other", inputs)
    other.AddShapeSys("other_shape", hf.Constraint.Poisson, "other_rel_uncertainty", inputs)
    channel.AddSample(signal)
    channel.AddSample(background)
    channel.AddSample(other)
    free = hf.Sample("free", "free", inputs)
    free.AddShapeFactor("free_shape")
    channel.AddSample(free)
    measurement.AddChannel(channel)
    measurement.CollectHistograms()
    workspace = hf.MakeModelAndMeasurementFast(measurement)
    path = str(tmp_path / "workspace.root")
    workspace.writeToFile(path)
    return path


@pytest.fixture
def workspace_file(tmp_path):
    return make_workspace(tmp_path, ["Lumi"])


def test_workspace_to_pyhf(workspace_file):
    hs3 = json.loads(RooWorkspace_to_HS3.export(workspace_file))
    ws = HS3_to_pyhf.convert(hs3)
    assert ws["observations"] == [{"name": "SR", "data": [135.0, 262.0]}]
    samples = {s["name"]: s for s in ws["channels"][0]["samples"]}
    assert samples["signal"]["data"] == [10.0, 15.0]
    # expression
    assert samples["signal"]["modifiers"] == [{"name": "kappa2", "type": "purefunc"}]
    assert ws["bindings"] == [{"names": ["kappa2"], "expression": "kappa ** 2", "language": "sympy"}]
    modifiers = {m["name"]: m for m in samples["background"]["modifiers"]}
    assert sorted(modifiers) == ["bkg_norm", "staterror_SR"]
    assert modifiers["bkg_norm"]["data"] == pytest.approx({"hi": 1.1, "lo": 0.9})
    assert modifiers["staterror_SR"]["data"] == pytest.approx([10.0, 0.0])  # second bin pruned
    # ShapeSys
    (shapesys,) = samples["other"]["modifiers"]
    assert shapesys["type"] == "shapesys"
    assert shapesys["data"] == pytest.approx([0.2 * 20.0, 0.1 * 40.0])
    assert samples["free"]["modifiers"] == [{"name": "free_shape", "type": "shapefactor", "data": None}]
    assert "lumi" not in [p["name"] for p in ws["measurements"][0]["config"]["parameters"]]

    pyhf.set_backend("jax")
    config = ws["measurements"][0]["config"]
    spec = {"channels": ws["channels"], "parameters": config["parameters"], "bindings": ws["bindings"]}
    model = pyhf.Model(spec, modifier_set=purefunc.enable(), poi_name=config["poi"], validate=False)
    assert model.config.poi_name == "kappa"
    assert model.config.param_set("staterror_SR").sigmas[0] == pytest.approx(0.1)
    pars = model.config.suggested_init()
    assert model.expected_actualdata(pyhf.tensorlib.astensor(pars)) == pytest.approx([135.0, 262.0])
    pars[model.config.par_slice("kappa").start] = 2.0
    assert model.expected_actualdata(pyhf.tensorlib.astensor(pars)) == pytest.approx([165.0, 307.0])


def test_constant_parameters(tmp_path):
    # free Lumi with its constraint, constant nuisance parameter
    hs3 = json.loads(RooWorkspace_to_HS3.export(make_workspace(tmp_path, ["alpha_bkg_norm"])))
    ws = HS3_to_pyhf.convert(hs3)
    parameters = {p["name"]: p for p in ws["measurements"][0]["config"]["parameters"]}
    assert parameters["bkg_norm"]["fixed"] and parameters["bkg_norm"]["inits"] == [0.0]
    assert not parameters["lumi"]["fixed"]
    assert parameters["lumi"]["auxdata"] == [1.0]
    assert parameters["lumi"]["sigmas"] == pytest.approx([0.02])
    modifiers = {s["name"]: [m["name"] for m in s["modifiers"]] for s in ws["channels"][0]["samples"]}
    assert "lumi" in modifiers["background"]


def test_workspace_name(workspace_file):
    with pytest.raises(ValueError, match="no RooWorkspace named"):
        RooWorkspace_to_HS3.export(workspace_file, "missing")
