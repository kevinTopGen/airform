#!/usr/bin/env python3
"""Generate a local HTML gallery for doctor_bank so pairs are easy to browse."""
from pathlib import Path
import json
import webbrowser
import os

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "doctor_bank"


def main() -> None:
    rows = []
    for doc in sorted(BANK.iterdir()):
        if not doc.is_dir():
            continue
        dj = doc / "doctor.json"
        if not dj.exists():
            continue
        info = json.loads(dj.read_text(encoding="utf-8"))
        for angle in ("front", "side"):
            adir = doc / angle
            if not adir.exists():
                continue
            for case in sorted(adir.iterdir()):
                if not case.is_dir():
                    continue
                b = case / "before.jpg"
                a = case / "after.jpg"
                if b.exists() and a.exists():
                    rows.append(
                        {
                            "doctor": info["name"],
                            "slug": doc.name,
                            "angle": angle,
                            "case": case.name,
                            "before": b.relative_to(ROOT).as_posix(),
                            "after": a.relative_to(ROOT).as_posix(),
                        }
                    )

    css = """
body{font-family:system-ui,sans-serif;margin:24px;background:#0f1115;color:#eee}
h1{margin:0 0 8px} .meta{opacity:.7;margin-bottom:24px}
.doctor{margin:32px 0 12px;padding-bottom:8px;border-bottom:1px solid #333}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
.card{background:#1a1d24;border-radius:12px;padding:12px;border:1px solid #2a2f3a}
.card h3{margin:0 0 8px;font-size:14px;font-weight:600}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.pair figure{margin:0}
.pair img{width:100%;height:220px;object-fit:cover;border-radius:8px;background:#000}
.pair figcaption{font-size:11px;opacity:.6;margin-top:4px;text-align:center}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;background:#2d6cdf;margin-left:6px}
.badge.side{background:#1f8a5b}
"""
    parts = [
        "<!doctype html><html><head><meta charset=utf-8>",
        "<title>Doctor Bank Gallery</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Doctor Rhinoplasty Bank</h1>",
        f"<p class=meta>{len(rows)} same-angle before/after pairs · front or side only</p>",
        f"<p class=meta>Folder: <code>{BANK}</code></p>",
    ]
    cur = None
    for r in rows:
        if r["slug"] != cur:
            if cur is not None:
                parts.append("</div>")
            parts.append(f'<h2 class="doctor">{r["doctor"]}</h2><div class="grid">')
            cur = r["slug"]
        badge = r["angle"]
        parts.append(
            f'<div class="card"><h3>{r["case"]} '
            f'<span class="badge {badge}">{r["angle"]}</span></h3>'
            f'<div class="pair">'
            f'<figure><img loading="lazy" src="{r["before"]}"><figcaption>before</figcaption></figure>'
            f'<figure><img loading="lazy" src="{r["after"]}"><figcaption>after</figcaption></figure>'
            f"</div></div>"
        )
    if cur:
        parts.append("</div>")
    parts.append("</body></html>")

    out = ROOT / "doctor_bank_gallery.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"pairs={len(rows)}")
    print(f"gallery={out}")
    print(f"bank={BANK}")
    # open in default browser
    webbrowser.open(out.as_uri())
    # open explorer
    if os.name == "nt":
        os.startfile(str(BANK))  # noqa: S606


if __name__ == "__main__":
    main()
