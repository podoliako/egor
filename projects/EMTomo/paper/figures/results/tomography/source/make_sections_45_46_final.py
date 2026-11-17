"""Clean separate 4.5/4.6 figures from the existing full-domain comparison data.

Run from projects/EMTomo:
    python -B paper/figures/results/tomography/source/make_sections_45_46_final.py
No resampling, new mask calculation, phase reversal or original-file overwrite.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from make_comparison import HERE, ROOT, PNG, draw_block, read, sha

PLANES = ("xy", "yz", "xz")
MODES = ("true", "soft", "hard")


def load_block(section):
    report_path = HERE / "selection_and_sections.json"
    report = read(report_path)
    block = next(b for b in report["blocks"] if b["section"] == section)
    block["extent_m"] = np.asarray(block["extent_m"])
    block["native_to_common_translation_m"] = np.asarray(block["native_to_common_translation_m"])
    data_path = HERE / f"section_{section.replace('.', '_')}_slices.npz"
    with np.load(data_path, allow_pickle=False) as arrays:
        block["slices"] = {f"{mode}_{plane}": arrays[f"{mode}_{plane}"].copy()
                           for mode in MODES for plane in PLANES}
        block["display_masks"] = {p: arrays[p+"_display_mask"].copy() for p in PLANES}
        block["native_display_masks"] = {p: arrays[p+"_native_display_mask"].copy() for p in PLANES}
    for record in block["records"].values():
        if sha(ROOT / record["checkpoint"]) != record["checkpoint_sha256"]:
            raise ValueError("Displayed checkpoint has changed")
    return block, report_path, data_path


def draw_clean(block):
    fig = plt.figure(figsize=(17, 10))
    spec = fig.add_gridspec(1, 1, left=.075, right=.95, top=.87, bottom=.14)[0, 0]
    draw_block(fig, spec, block, "d" if block["section"] == "4.5" else "e")
    for text in list(fig.texts):
        if not text.get_text().startswith(("True", "Soft", "Hard")):
            text.remove()
    return fig


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.linewidth": .5})
    for section in ("4.5", "4.6"):
        block, report_path, data_path = load_block(section)
        stem = f"section_{section.replace('.', '_')}_final"
        fig = draw_clean(block)
        fig.savefig(PNG / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=.12)
        fig.savefig(HERE / f"{stem}.pdf", dpi=220, bbox_inches="tight", pad_inches=.12)
        plt.close(fig)
        provenance = dict(
            section=section, source="Existing full-domain comparison, exact saved slice and mask arrays",
            title=False, bottom_notes=False, rows=list(MODES), columns=list(PLANES),
            native_cuts_xyz_m=[block["section_x_native_m"],block["section_y_native_m"],block["section_z_native_m"]],
            coordinate_offset_m=block["native_to_common_translation_m"].tolist(),
            color_limits_km_s=block["color_limits_km_s"], no_phase_reversal=True,
            fault_geometry=block.get("fault_geometry"), sensitivity_envelopes=block["sensitivity_envelopes"],
            mask_caveat="Display sensitivity envelope, not a validated resolution region; pre-final-update diagnostics",
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                           (Path(__file__).resolve(),HERE/"make_comparison.py",report_path,data_path)},
        )
        (HERE/f"{stem}.json").write_text(json.dumps(provenance,indent=2,allow_nan=False)+"\n")
        print(PNG/f"{stem}.png")


if __name__ == "__main__":
    main()
