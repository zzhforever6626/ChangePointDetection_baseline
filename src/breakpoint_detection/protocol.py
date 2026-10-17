"""Published settings and explicitly disclosed source adapters."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SETTINGS = json.loads((PACKAGE / "baseline_settings.json").read_text(encoding="utf-8"))
MARGINS = tuple(SETTINGS["margins"])
if SETTINGS["metric"] != "event_micro_f1":
    raise ValueError("The active 2021 protocol requires event micro-F1")
REFERENCE_F1 = {
    "tscp2": {24: .6400, 50: .8104, 75: .8428},
    "rulsif": {24: .3175, 50: .3317, 75: .3700},
    "floss": {24: .2083, 50: .3375, 75: .4233},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {p.relative_to(PACKAGE).as_posix(): sha256(p)
            for p in sorted(PACKAGE.rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def empty_output(path):
    output = Path(path).resolve()
    official = PACKAGE.parents[1] / "results/yahoo-a4"
    if output == official or official in output.parents:
        raise ValueError("run output must not overwrite official results; use reproductions/")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("choose a new or empty output directory; existing results are not overwritten")
    return output
