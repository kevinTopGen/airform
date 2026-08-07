# Airform Synchronization Contract

Status: audited integration seam for the tournament subsystem. This document
records what exists in Airform now and separates it from proposed merge work.

## Audited host snapshot

- Repository: `https://github.com/kevinTopGen/airform`
- Commit: `3ae2ddd78a248b53cbcc8c0bf00ee7e03ea1b7bd`
- Commit timestamp: `2026-08-06T19:08:10-04:00`
- Audit date: August 6, 2026

At this snapshot Airform is a Python simulation library and command-line demo.
It has no frontend, HTTP API, database, or scraper. Its README explicitly lists
"Frontend, API, scraper" as not built. Consequently, there is no current Airform
page structure, component library, route, or API endpoint for the tournament to
copy. Visual synchronization must occur after the contributor adds a frontend;
until then, the safe shared seam is data identity.

The authoritative audited files are:

- `nosesim/contracts.py`: `SurgeonSignature`, `NoseParams`, and JSON serialization;
- `nosesim/signatures.py`: signature fitting and the placeholder archetype registry; and
- `README.md`: current capabilities and explicit unbuilt surfaces.

## Source ownership and the join

Airform and the scraper own different facts. Neither source should be copied
over or silently replaced by the tournament.

| Fact | Authority | Tournament use |
| --- | --- | --- |
| Surgeon identity and simulation signature | Airform `SurgeonSignature` JSON | Adapt `id` and `name` into `Surgeon` |
| Before/after case IDs and image links | Scraper `doctor.json` plus `manifest.json` | Adapt eligible manifest entries into `SurgeryCase` |
| Matchups, votes, case ratings, surgeon scores | Tournament | Persist only stable external IDs |

The non-negotiable identity invariant is:

```text
SurgeryCase.surgeonId === Surgeon.id === SurgeonSignature.id
```

`SurgeonSignature.id` passes through unchanged. Do not derive it from the
display name, create a tournament-only surgeon ID, or remap it in UI code.

The current scraper folders use `doctor.json.slug` as their surgeon key and
`manifest.json.sets[].set_id` as their case key. When real signatures are fit,
Airform's `fit(surgeon_id=...)` must receive that exact doctor slug. For example,
a case loaded from `rhinoplasty_dataset/zhuravsky_ruslan/manifest.json` uses
`surgeonId: "zhuravsky_ruslan"`, so the corresponding signature must use
`id: "zhuravsky_ruslan"`.

Airform's current `ARCHETYPES` IDs (`conservative`, `alar`, `dorsal`, `tip`, and
`signature`) do not match the real scraper doctor slugs. They are demo-only
simulation choices, not identities to attach to scraped cases or tournament
scores.

## Airform JSON to tournament mapping

`SurgeonSignature.to_dict()` emits this stable JSON surface:

```ts
interface AirformSurgeonSignatureJson {
  id: string;
  name: string;
  tagline: string;
  n_pairs: number;
  delta: Partial<Record<AirformNoseField, number>>;
  std: Partial<Record<AirformNoseField, number>>;
  delta_modes: Partial<Record<AirformNoseField, "absolute" | "proportional">>;
}
```

The implementation is in
`src/tournament/integration/airformContracts.ts`.

| Airform field | Tournament mapping |
| --- | --- |
| `id` | `Surgeon.id`, unchanged |
| `name` | `Surgeon.name`, unchanged |
| `tagline` | Not part of the tournament domain; Airform retains it |
| `n_pairs` | Not a vote/case count; Airform retains it |
| `delta`, `std`, `delta_modes` | Simulation-only; Airform retains them |

`n_pairs` is the count of usable pairs accepted by Airform's fitting pipeline,
not the count of tournament-eligible cases. It must not affect tournament score
confidence or eligibility.

The scraper case manifest remains authoritative for every `SurgeryCase` image
link. The tournament must not infer image paths from a signature, folder name,
or surgeon ID. It should adapt each eligible manifest entry through the existing
`SurgeryCase` boundary using:

- `id = manifest.sets[].set_id`;
- `surgeonId = doctor.slug` (which must equal the Airform signature ID);
- `procedure = "rhinoplasty"` for this dataset;
- `beforeImageUrl` and `afterImageUrl` from the manifest's selected display
  references; and
- `active = manifest.sets[].ok` plus any later consent/publication gate.

The current manifests include both remote source URLs and local downloaded
paths. Which pair becomes browser-displayable URLs is a deployment concern for
the future host adapter; the manifest remains the provenance source either way.

## Concrete merge bridge

### Available now, without an Airform server

1. Export the Airform registry by serializing each `SurgeonSignature.to_dict()`
   into a JSON artifact.
2. Read that artifact at the application composition boundary and call
   `adaptAirformSurgeonSignature` for each record.
3. Independently load `doctor.json` and `manifest.json`, adapting eligible sets
   through `adaptSurgeryCase`.
4. Before enabling the tournament, fail startup/build validation if any case's
   `surgeonId` does not resolve to an adapted Airform surgeon.
5. Register the resulting provider with `TournamentService`. Keep Airform
   simulation statistics out of matchup, voting, and rating storage.

This static export path is the smallest merge surface supported by the audited
host. It requires no invented backend.

### Proposed future service bridge (not present in Airform)

If the contributor adds a backend, the following are **proposed endpoints**, not
existing Airform routes:

- **PROPOSED** `GET /api/surgeon-signatures`: return serialized
  `SurgeonSignature.to_dict()` records.
- **PROPOSED** `GET /api/tournament/cases`: return authorized, manifest-backed
  `SurgeryCase` records without patient identity.
- **PROPOSED** `GET /api/tournament/matchup`,
  `POST /api/tournament/vote`, and
  `GET /api/tournament/surgeons/:id/score`: mount the tournament's already
  specified route contracts in the host service.

The route handlers should compose three existing boundaries:

```text
Airform signature JSON -> adaptAirformSurgeonSignature -> Surgeon
scraper manifests       -> adaptSurgeryCase             -> SurgeonCaseProvider
SurgeonCaseProvider     -> TournamentService            -> proposed HTTP routes/UI
```

No tournament module should import Airform Python internals or read scraper
directories directly. Those operations belong in host-side export/provider
adapters and application wiring.

## UI synchronization

There is no Airform frontend at the audited commit, so matching its current UI
is impossible. When a frontend lands, synchronize at the host composition layer:

- use the host shell, typography, color tokens, spacing, buttons, and responsive
  breakpoints around the tournament components;
- mount the tournament under a host-owned route and navigation label;
- place `SurgeonCommunityScore` on the host profile using the unchanged
  `SurgeonSignature.id`; and
- keep the comparison screen anonymous until a vote is accepted.

Do not fork tournament rating behavior merely to match presentation. Host UI
components may wrap the domain/service layer without changing its contracts.

## Acceptance checks

- A serialized Airform signature adapts to `{ id, name }` without changing `id`.
- Every eligible manifest case resolves a signature by exact ID equality.
- A missing signature or duplicate ID fails integration validation; it never
  creates a placeholder surgeon.
- Manifest image references, case IDs, and `ok` state survive adaptation.
- Signature statistics never enter tournament vote/rating calculations.
- Matchup responses expose no surgeon ID or name before voting.
- The real host frontend is re-audited before claiming visual conformance.
- Any introduced endpoint is labeled proposed until it exists and is tested in
  Airform.

## Known risks

- **Identity mismatch:** current Airform archetypes and real scraper doctor slugs
  are different namespaces. Real signatures must be fit with scraper slugs.
- **No integration runtime yet:** Airform has no API or frontend, so static export
  is the only audited bridge today.
- **Image delivery:** manifest source URLs may expire, block hotlinking, or be
  unsuitable for public use; local paths are not browser URLs.
- **Eligibility and rights:** `ok` means the scraper downloaded a pair; it does
  not prove consent, publication rights, correct pairing, or tournament fitness.
- **Clinical/reputation claims:** Airform deliberately uses unnamed archetypes
  because attaching predicted outcomes to real practices creates defamation and
  medical-claims risk. Real-name publication requires an explicit product/legal
  decision outside this adapter.
- **Measurement maturity:** the audited README warns that FaceMesh cannot
  reliably measure nose width and that real signatures are not yet built. A
  score can rank image preferences, but it must not validate simulation accuracy.
- **Contract drift:** snake_case serialization is audited only at the commit
  above. Re-audit `SurgeonSignature.to_dict()` when updating the host revision.
