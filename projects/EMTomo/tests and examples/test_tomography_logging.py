"""Tests for descriptive, date-groupable tomography run identifiers."""

import json
import re

from tomography.tomography_logging import TomographyLogger


def test_logger_uses_descriptive_run_id_and_persists_identity(tmp_path):
    logger = TomographyLogger(
        base_dir=tmp_path,
        run_name="EM release",
        run_version="1.2.3",
        run_tags={"topn": 3, "lam": 0.01},
    )
    logger.save_meta({}, [], [])

    assert re.fullmatch(
        r"run_em-release_v1\.2\.3_topn-3_lam-0\.01_\d{8}_\d{6}", logger.run_id
    )

    meta = json.loads((logger.run_dir / "meta.json").read_text())
    assert meta["run_name"] == "em-release"
    assert meta["run_version"] == "1.2.3"
    assert meta["run_tags"] == {"topn": "3", "lam": "0.01"}
    assert "started_at" in meta
