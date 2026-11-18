# statmodel-converters

Converters between formats for HistFactory-style statistical models:

| Script | From | To |
|---|---|---|
| `RooWorkspace_to_HS3.py` | RooWorkspace in a ROOT file | [HS3](https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard) JSON |
| `HS3_to_pyhf.py` | HS3 JSON | [pyhf](https://github.com/scikit-hep/pyhf) JSON |
| `RooWorkspace_to_pyhf.py` | RooWorkspace in a ROOT file | pyhf JSON (both steps in one) |

Expressions (`RooFormulaVar` scaling a sample) are written as `purefunc` modifiers with `bindings`,
following [pyhf PR #2717](https://github.com/scikit-hep/pyhf/pull/2717).

## Setup

```
pixi install
pixi run test
```

The environment uses the nightly ROOT wheels built from ROOT master and pyhf from the PR branch.
`HS3_to_pyhf.py` itself only needs the Python standard library.

The nightly wheel is replaced in place, so `pixi.lock` is not tracked. CI runs the tests every night
against the latest wheel, to catch changes in the HS3 export of ROOT.

## Usage

```
pixi run python RooWorkspace_to_HS3.py workspace.root model_hs3.json
pixi run python HS3_to_pyhf.py model_hs3.json model_pyhf.json
pixi run python RooWorkspace_to_pyhf.py workspace.root model_pyhf.json
```

Options: `--workspace` (name, if the file has several), `--data` (dataset, default `obsData`),
`--poi` (default: first POI in the model), `--fix-other-pois`.

Building the model in pyhf needs the jax backend and the `purefunc` modifier set. `Workspace.model()`
does not forward `bindings`, so build the model directly:

```python
import json
import pyhf
from pyhf.contrib.extended_modifiers import purefunc

pyhf.set_backend("jax", "scipy")
ws = pyhf.Workspace(json.load(open("model_pyhf.json")), validate=False)
config = ws.get_measurement()["config"]
model = pyhf.Model(
    {"channels": ws["channels"], "parameters": config["parameters"], "bindings": ws["bindings"]},
    modifier_set=purefunc.enable(),
    poi_name=config["poi"],
    validate=False,
    modifier_settings={"normsys": {"interpcode": "code4"}, "histosys": {"interpcode": "code4p"}},
)
data = ws.data(model)
```

The interpolation codes are not part of the pyhf JSON. `code4` / `code4p` match the HistFactory default,
and the conversion fails for models using anything else.

## How HS3 maps to pyhf

| HS3 | pyhf |
|---|---|
| `normfactor` | `normfactor`; init, bounds and `fixed` from `default_values` and `domains` |
| `normfactor` named `Lumi` | dropped if constant at 1, otherwise `lumi` with the width of its constraint |
| `normsys`, `histosys` | same type, named after the parameter without `alpha_` |
| constant or shifted nuisance parameter | `fixed` and `inits` in the measurement |
| `staterror` | `staterror_<channel>` with the uncertainties of each sample; pyhf combines them in quadrature |
| `shapefactor` without constraints | `shapefactor` |
| `shapefactor` with Poisson constraints | `shapesys`, uncertainty = yield / sqrt(tau) |
| `shapefactor` with Gaussian constraints | `staterror` with a name unique to the sample |
| constant gamma parameter (pruned bin) | zero uncertainty in that bin, pyhf then fixes the parameter |
| `custom` with a `generic` function | `purefunc` and a binding with the expression in sympy syntax |
| `custom` with a `step` function (one expression per bin) | sample split into one sample per bin, each with its own `purefunc` |

## Limitations

- `purefunc` evaluation loops over bindings x samples x channels in Python. Models with hundreds of
  bindings take around a minute per evaluation.
- Splitting samples per bin gives each split sample its own `shapesys`, which adds fixed placeholder
  parameters.
- pyhf constrains `staterror` with a Gaussian, HistFactory typically with a Poisson. Widths agree.
- Expressions must be valid Python syntax after translating `^` and `pow(a, b)` to `**`. `?:`, `&&`,
  `TMath::` calls and functions outside a short list are rejected.
- Parameters named like sympy objects (`beta`, `gamma`, `E`, `I`, ...) are misread by the `purefunc`
  parser.
- The HS3 export in ROOT master is changing. Files written by other ROOT versions may not convert.
