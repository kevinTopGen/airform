# Rhinoplasty Tournament Subsystem — Integration Specification

Status: merge-ready subsystem contract for the hackathon build.

## 1. Context and integration posture

The hackathon application will eventually combine two independently developed areas:

1. a primary surgeon-specific visualization experience; and
2. this anonymous, pairwise before/after tournament.

The other developer's application is authoritative for surgeon profiles, surgery cases, image storage, image processing, consent status, and any clinical metadata. Its framework, database, routes, and field names are intentionally treated as unknown.

The tournament is a detachable subsystem. It consumes a small normalized view of the authoritative data, then owns matchups, votes, ratings, and score presentation. It must not require the host repository to adopt the tournament's model for surgeons or images.

The integration rule is:

> Share only `Surgeon` and `SurgeryCase` contracts. Persist their IDs at the boundary. Replace one `SurgeonCaseProvider` adapter when the host repository is available; do not rewrite tournament logic.

## 2. Ownership boundary

| Concern | Authoritative owner |
| --- | --- |
| Surgeon identity, profile, and public visibility | Host application |
| Surgery case identity, procedure, eligibility, and consent | Host application |
| Before/after image locations and delivery policy | Host application |
| Computer-vision or image-generation workflow | Host application |
| Tournament matchup selection | Tournament subsystem |
| Anonymous vote validation and persistence | Tournament subsystem |
| Case ratings and surgeon score aggregation | Tournament subsystem |
| Tournament UI, leaderboard, and reusable score display | Tournament subsystem |

The tournament must never write to host-owned surgeon, case, or image records. Deleting or hiding a host case makes it ineligible for new matchups, but existing tournament records retain only its stable ID and audit data.

## 3. Shared contracts

The adapter returns normalized, read-only objects. A host integration may have many more fields; the tournament cannot depend on them.

```ts
export interface Surgeon {
  id: string;
  name: string;
  slug?: string;
  profileImageUrl?: string;
}

export interface SurgeryCase {
  id: string;
  surgeonId: string;
  procedure: string;
  beforeImageUrl: string;
  afterImageUrl: string;
  view?: string;
  active?: boolean;
  metadata?: Record<string, unknown>;
}
```

Contract rules:

- `Surgeon.id` and `SurgeryCase.id` are opaque, stable values. Tournament code must not parse meaning from them.
- The adapter serializes numeric or UUID host keys to strings without inventing a second identity.
- `SurgeryCase.surgeonId` must equal an existing authoritative `Surgeon.id`.
- Image URLs may be short-lived signed URLs. They are display references, not persistent tournament data.
- `active: false` always excludes a case. The provider also applies the host application's publication and consent rules before returning a case from `listEligibleCases`.
- `metadata` is an escape hatch for adapter-specific context. Core tournament behavior cannot require or persist it.
- Additional host fields are ignored unless this specification is deliberately revised.
- A future host schema change should be absorbed inside the adapter.

## 4. Provider seam

Tournament services depend only on this interface:

```ts
export interface SurgeonCaseProvider {
  getSurgeon(surgeonId: string): Promise<Surgeon | null>;
  listSurgeons(): Promise<readonly Surgeon[]>;
  getCase(caseId: string): Promise<SurgeryCase | null>;
  listEligibleCases(input?: {
    procedure?: string;
    limit?: number;
  }): Promise<readonly SurgeryCase[]>;
}
```

During independent development, an in-memory or fixture-backed implementation supplies the dataset. After merge, a host adapter implements the same interface using the host repository or service. The application composition root changes which implementation is registered. Matchmaking, voting, scoring, API handlers, and UI components must not import the fixture adapter or the host data layer directly.

```text
fixture data ----> FixtureSurgeonCaseProvider --\
                                              +--> tournament services
host data -------> HostSurgeonCaseProvider ----/
```

## 5. Tournament-owned domain

### Matchup

A matchup contains two distinct eligible case IDs, normally from different surgeons, for the same requested procedure. Its public response includes before/after image URLs but never reveals surgeon identity before a vote.

Required fields:

- `id`
- `leftCaseId`
- `rightCaseId`
- `procedure`
- `status`: `open`, `voted`, or `expired`
- `createdAt`
- `expiresAt`

### Vote

A vote records one preference in one matchup. The server validates that the selected case belongs to that matchup and accepts no more than one vote per matchup and voter/session combination.

Required fields:

- `id`
- `matchupId`
- `selectedCaseId`
- `rejectedCaseId`
- `anonymousVoterId` or non-identifying session hash
- `idempotencyKey`
- `createdAt`

### Rating and surgeon score

Each accepted vote updates the two case ratings using one deterministic pairwise-rating policy. Elo is the baseline hackathon policy:

- new cases start at `1500`;
- expected score uses the standard Elo formula;
- accepted winner/loser results use a fixed `K = 32` during the hackathon;
- updates occur atomically with vote creation; and
- algorithm name and version are stored with rating snapshots so the policy can be recalculated later.

A surgeon's displayed score is derived from that surgeon's rated, tournament-eligible cases. For the initial implementation, use the mean current case rating and expose the number of cases and accepted votes with it. The UI must label it as a community preference score, not a medical-quality score.

Do not display a ranked score until the surgeon meets the configured minimum sample threshold. The initial threshold is three distinct rated cases and ten accepted votes across those cases. Below the threshold, return `provisional: true` and show "Not enough votes yet."

## 6. Persistence contract

The standalone hackathon implementation uses an in-memory repository with browser persistence so it can run before the host backend is known. Tournament services depend on a tournament repository seam, not directly on browser storage. A production merge replaces that repository with a server/database implementation without changing matchmaking, voting, or score calculation.

Use the same logical names for demo browser-storage keys and future production tables or collections. The `tournament_` prefix prevents collisions during merge:

| Name | Purpose | Cross-system fields |
| --- | --- | --- |
| `tournament_matchups` | Pair offered to a voter | `left_case_id`, `right_case_id` |
| `tournament_votes` | One validated preference | `selected_case_id`, `rejected_case_id` |
| `tournament_case_ratings` | Current and versioned case rating | `case_id` |
| `tournament_surgeon_scores` | Rebuildable aggregate/cache | `surgeon_id` |

Persistence rules:

- Store host IDs as strings under the names above; do not copy host profile or case records.
- Do not create database foreign keys into host-owned tables. Validate references through `SurgeonCaseProvider` so the subsystem remains portable.
- Enforce uniqueness for `(matchup_id, anonymous_voter_id)` and for `idempotency_key` in the repository. Production storage should back both rules with database constraints.
- Store no patient identity, source filenames, facial landmarks, generated images, diagnoses, or clinical notes.
- Treat score rows as derived data. They must be rebuildable from accepted votes and the current provider view.
- A vote and both case-rating changes are one repository operation. Production storage must commit them in one transaction; demo storage must publish the updated snapshot only after the complete operation succeeds.

## 7. HTTP API contract

Routes may be mounted under a host prefix, but request and response semantics remain stable.

### `GET /api/tournament/matchup`

Query parameters:

- `procedure` (optional; defaults to `rhinoplasty`)

Success, `200`:

```json
{
  "matchupId": "match_01",
  "expiresAt": "2026-08-06T20:00:00.000Z",
  "left": {
    "caseId": "case_101",
    "beforeImageUrl": "https://images.example/before-101",
    "afterImageUrl": "https://images.example/after-101"
  },
  "right": {
    "caseId": "case_205",
    "beforeImageUrl": "https://images.example/before-205",
    "afterImageUrl": "https://images.example/after-205"
  }
}
```

The response must not include surgeon IDs, surgeon names, source paths, ratings, or rank. Return `404` with code `NO_ELIGIBLE_MATCHUP` when a valid pair cannot be formed.

### `POST /api/tournament/vote`

Request:

```json
{
  "matchupId": "match_01",
  "selectedCaseId": "case_205",
  "idempotencyKey": "9e07237a-1f96-4ba9-9214-8f30f3f37fc5"
}
```

Success, `201` for a new vote or `200` for an idempotent replay:

```json
{
  "voteId": "vote_01",
  "accepted": true,
  "nextMatchupHref": "/api/tournament/matchup?procedure=rhinoplasty"
}
```

Reject an expired or already-voted matchup with `409`; reject a case that is not part of the matchup with `422`; reject malformed input with `400`. The server derives the anonymous voter/session identifier rather than trusting a client-provided identity.

### `GET /api/tournament/surgeons/:id/score`

Success, `200`:

```json
{
  "surgeonId": "surgeon_17",
  "procedure": "rhinoplasty",
  "score": 1548,
  "ratedCaseCount": 6,
  "voteCount": 83,
  "provisional": false,
  "algorithm": "elo-case-mean-v1",
  "updatedAt": "2026-08-06T19:30:00.000Z"
}
```

Return `404` when the provider has no such surgeon. A valid surgeon with insufficient data still returns `200`, with `score: null` and `provisional: true`.

### `GET /api/tournament/leaderboard`

Query parameters:

- `procedure` (optional; defaults to `rhinoplasty`)
- `limit` (optional; server-capped)

The response includes only non-provisional surgeon scores and enough host-provided display information to render links. This endpoint is optional for the first hackathon demo; matchup, voting, and surgeon score endpoints are required.

## 8. UI contract

The required tournament screen:

- presents two side-by-side before/after case pairs;
- uses neutral labels such as "Result A" and "Result B";
- hides surgeon identity, score, and ranking until after vote acceptance;
- has clear keyboard and pointer controls for choosing either result;
- disables repeat submission while a vote is pending;
- displays recoverable loading, empty, expired-matchup, and error states; and
- requests the next matchup after a successful vote.

The reusable surgeon score component accepts `surgeonId` and `procedure`, reads the score endpoint, and renders the score, sample size, provisional state, algorithm label, and disclaimer. It must not require the tournament screen or fixture dataset.

## 9. Security, privacy, and medical disclaimer

- Only cases with documented rights, patient consent, and tournament eligibility may be exposed by the host provider.
- Images must be de-identified. The tournament must not accept, infer, or display patient names or other protected health information.
- Do not log image URLs, face embeddings, or source metadata. URLs should follow the host's authorization and expiry policy.
- Add reasonable abuse controls to voting, such as session limits, idempotency, and rate limiting. Do not fingerprint users beyond what is necessary for basic vote integrity.
- Scores measure anonymous community aesthetic preference within the available sample. They do not measure surgical safety, clinical quality, likely patient outcome, or professional competence.
- This experience is informational and experimental, is not medical advice, and must not be used to choose treatment without consultation with a qualified clinician.

Recommended visible copy:

> Community preference scores reflect anonymous votes on selected before/after images. They are not measures of medical quality, safety, or expected results and do not constitute medical advice.

## 10. Non-goals

The tournament subsystem does not own or implement:

- the host application's surgeon directory or profile database;
- case ingestion, moderation, consent capture, or image storage;
- OpenCV, FaceMesh, generative imaging, or visualization logic;
- patient accounts, clinical recommendations, booking, or treatment matching;
- claims that one surgeon is medically superior to another;
- cross-repository database migrations for host-owned data; or
- synchronization by copying the full host surgeon/case dataset into tournament tables.

## 11. Definition of done

The subsystem is ready to merge when all of the following are true:

- [ ] `Surgeon` and `SurgeryCase` are the only shared data contracts.
- [ ] All tournament services receive data through `SurgeonCaseProvider`.
- [ ] A fixture provider can run the feature without the future host repository.
- [ ] A provider contract test can be reused against the future host adapter.
- [ ] Eligible cases produce anonymous, same-procedure, cross-surgeon matchups.
- [ ] Ineligible and missing cases never produce a new matchup.
- [ ] The vote API validates membership, expiry, idempotency, and one-vote constraints.
- [ ] Demo state uses the in-memory/browser repository, and a production repository can replace it without changing domain logic.
- [ ] Vote persistence and rating updates are atomic and deterministic at the repository boundary.
- [ ] Surgeon scores are derived from case ratings and include sample counts/provisional state.
- [ ] The tournament screen supports success, pending, empty, expired, and error paths.
- [ ] The profile score component works from only `surgeonId` and `procedure`.
- [ ] Tournament storage uses the `tournament_` namespace and stores only host IDs.
- [ ] No patient identity or clinical metadata enters tournament storage or logs.
- [ ] Automated tests cover adapter conformance, matchup anonymity, vote validation, and score calculation.
- [ ] Visible privacy and medical-disclaimer copy is present.
- [ ] Merging into the host repository requires replacing the provider adapter and mounting the subsystem, not rewriting its domain logic.
