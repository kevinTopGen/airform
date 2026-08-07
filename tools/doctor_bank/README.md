# Doctor-bank tooling

This directory preserves the scraper and quality-control source from the
original hackathon workspace inside the canonical `airform` repository.

Generated photographs remain in the repository-root `doctor_bank/` directory,
which is intentionally ignored. The local HTML gallery, rejected cases,
`tmp_qc/`, and Python caches are also ignored.

## Commands

```powershell
python tools/doctor_bank/build_doctor_bank.py
python tools/doctor_bank/strict_filter_bank.py
python tools/doctor_bank/make_gallery.py
```

`build_doctor_bank.py` rebuilds doctor directories and
`strict_filter_bank.py` rewrites and renumbers the bank in place. Do not run
either command against another contributor's active download. Copy the bank to
a disposable workspace before tuning QC rules.

The React demo consumes fictional cases. The fitted numeric surgeon signatures
under `data/signatures/` are the safe integration artifact for the preview
pipeline; real patient photographs are never committed.
