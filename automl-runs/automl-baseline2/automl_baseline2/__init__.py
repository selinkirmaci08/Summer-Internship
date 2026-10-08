"""AutoML baseline v2 -- a protocol-matched reference for one agentic ML run.

The v1 package (`../automl-baseline`) answers "what does off-the-shelf AutoML reach
on this dataset under *its own* honest protocol". That is a good number, but it is
not comparable row-for-row with an agentic run that chose a different split, a
different horizon grid and a different set of trading days.

This package does not replace v1 -- it imports it. `automl_baseline.features`,
`automl_baseline.metrics`, `automl_baseline.baselines` and `automl_baseline.search`
are reused unchanged. What is added here is the part v1 has no way to express:
pinning the evaluation to an agentic run's exact rows, so the only thing that
differs between the two systems is the modelling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.1.0"

# The v1 package lives in a sibling directory whose name has a hyphen, so it cannot
# be imported by name -- put the directory that *contains* it on the path instead.
_V1_DIR = Path(os.environ.get("AUTOML_BASELINE_V1", Path(__file__).resolve().parents[2] / "automl-baseline"))

if not (_V1_DIR / "automl_baseline").is_dir():
    raise ImportError(
        "cannot find the v1 package at {}. Set AUTOML_BASELINE_V1 to the directory "
        "containing 'automl_baseline/'.".format(_V1_DIR)
    )

if str(_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_DIR))

V1_DIR = _V1_DIR
