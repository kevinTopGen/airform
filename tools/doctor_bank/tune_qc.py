from pathlib import Path
from PIL import Image
from strict_filter_bank import features, is_worms_eye, classify_strict

tests = [
    "tmp_qc/audit/07_front_case_028/before.jpg",
    "tmp_qc/audit/13_front_case_008/before.jpg",
    "tmp_qc/audit/00_front_case_003/before.jpg",
    "tmp_qc/audit/08_side_case_027/before.jpg",
    "tmp_qc/audit/10_side_case_026/before.jpg",
    "tmp_qc/audit/02_side_case_007/before.jpg",
    "tmp_qc/audit/01_side_case_029/before.jpg",
]
for p in tests:
    im = Image.open(Path("..") / p if not Path(p).exists() else p)
    # resolve from repo root
    path = Path(__file__).resolve().parents[2] / p
    im = Image.open(path)
    f = features(im)
    print(path.parent.name, "->", classify_strict(im))
    print(
        f"  off={f['off']:.3f} sym={f['sym']:.1f} e_imb={f['e_imb']:.2f} "
        f"er={f['edge_ratio']:.2f} dark={f['dark_frac']:.4f} midD={f['mid_darker']} "
        f"fm={f['fm_ratio']:.2f} cyn={f['cyn']:.2f} worms={is_worms_eye(f)}"
    )
