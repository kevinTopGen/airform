# Agent instructions (scope: this repository)

## Repository map

- `nosesim/` and `scripts/`: Python landmark measurement and geometric rendering pipeline.
- `src/`: React tournament application and reusable tournament subsystem.
- `src/tournament/integration/airformContracts.ts`: authoritative TypeScript adapter seam.
- `docs/`: integration contracts and merge guidance; read only when the task requires them.
- `public/fixtures/`: fictional demo images only. Never add real patient photographs here.

## Integration invariants

- Preserve `SurgeryCase.surgeonId === Surgeon.id === SurgeonSignature.id` across Python, scraper, API, and UI work.
- Treat scraper manifests as the source of case IDs and image links; do not duplicate that mapping in UI code.
- Keep the tournament replaceable through its provider, repository, matchmaker, and rating-engine interfaces.
- Do not claim proposed API endpoints already exist; check the live implementation first.

## Shared-repository workflow

- Before editing, run `git fetch origin --prune`, inspect `git status -sb`, and compare the current branch with its remote and `origin/main`.
- Work on a focused feature branch named `agent/<scope>`; never commit directly to `main`.
- Treat unrelated modified, staged, or untracked files as protected work owned by another contributor. Do not move, delete, revert, stage, or rewrite them.
- Commit each independently useful, passing slice atomically. Stage explicit paths when the tree contains more than one workstream.
- Use terse conventional commit messages such as `feat: add tournament rating engine` or `fix: resume open matchup after reload`.
- After each atomic commit, fetch again, integrate upstream changes without rewriting shared history, and push with `git push -u origin HEAD`.
- Never force-push a shared branch unless the user explicitly authorizes that exact operation.
- Report the pushed branch and commit hash so other contributors can synchronize immediately.

## Verification

- Tournament: `npm test` and `npm run build`.
- Python syntax: `python -m compileall -q nosesim scripts` plus any task-specific checks documented in `README.md`.
- Before committing: `git diff --check`; before pushing: confirm `git status -sb` contains only the intended slice.

## Protected data

- Real face photographs and derived biometric data must remain outside Git history, consistent with `.gitignore` and `README.md`.
- Do not force-add `rhinoplasty_dataset/`, `doctor_bank/`, `tmp_qc/`, or another scraper output directory.
- Do not alter another agent's scraper or dataset outputs without explicit ownership of that lane.
