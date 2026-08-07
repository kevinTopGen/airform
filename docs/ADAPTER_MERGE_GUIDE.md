# Tournament Adapter and Merge Guide

Use this guide when the other developer's repository becomes available. Their surgeon/case/image model remains authoritative. The merge should translate that model at one boundary and leave tournament behavior unchanged.

## Target result

```text
Host surgeon/case/image data
            |
            v
  HostSurgeonCaseProvider   <-- only integration-specific implementation
            |
            v
   SurgeonCaseProvider
            |
            +--> matchmaking
            +--> voting
            +--> ratings and surgeon scores
            +--> tournament and profile UI
```

The seam is identity, not duplicated data: tournament rows persist opaque `surgeonId` and `caseId` strings. The adapter resolves those IDs and obtains current, authorized image URLs when needed.

## Merge procedure

### 1. Locate the host source of truth

Identify the host application's:

- surgeon entity and stable primary key;
- surgery case entity and stable primary key;
- case-to-surgeon relationship;
- procedure label or enum;
- before/after image access method; and
- publication, moderation, and patient-consent eligibility rules.

Do not rename or migrate host tables just to match the tournament vocabulary.

### 2. Write one host adapter

Implement the `SurgeonCaseProvider` contract from [TOURNAMENT_INTEGRATION_SPEC.md](./TOURNAMENT_INTEGRATION_SPEC.md). Translate host fields here, including casing, enums, numeric IDs, image signing, and eligibility.

Example mapping:

```ts
export class HostSurgeonCaseProvider implements SurgeonCaseProvider {
  constructor(private readonly hostRepository: HostRepository) {}

  async getCase(caseId: string): Promise<SurgeryCase | null> {
    const source = await this.hostRepository.findPublishedCase(caseId);
    if (!source) return null;

    return {
      id: String(source.primaryKey),
      surgeonId: String(source.ownerSurgeonKey),
      procedure: normalizeProcedure(source.procedureCode),
      beforeImageUrl: await this.hostRepository.getDisplayUrl(source.beforeAsset),
      afterImageUrl: await this.hostRepository.getDisplayUrl(source.afterAsset),
      view: source.viewName,
      active: source.isPublished && source.hasTournamentConsent,
    };
  }

  // Implement getSurgeon, listSurgeons, and listEligibleCases the same way.
}
```

Keep host imports inside this adapter and its wiring module. Do not import host ORM models into matchmaking, ratings, route handlers, or components.

### 3. Normalize IDs without remapping them

- Serialize the host's stable key with `String(hostId)`.
- Do not generate separate tournament surgeon or case IDs.
- Do not infer relationships from filenames, folder names, slugs, or display names.
- If host IDs are not stable, add stable IDs on the host side before integrating.
- If two sources can produce the same textual ID, add a stable namespace in the adapter, such as `legacy:123`, and use it consistently forever.

Changing IDs after votes exist breaks score continuity. If an unavoidable ID migration occurs, provide an explicit old-to-new mapping and migrate every tournament reference in one controlled operation.

### 4. Replace provider registration

At the application composition root, replace the development provider:

```ts
const surgeonCaseProvider = new FixtureSurgeonCaseProvider(fixtures);
```

with the host adapter:

```ts
const surgeonCaseProvider = new HostSurgeonCaseProvider(hostRepository);
```

There should be no other tournament code change. If replacing the provider causes changes throughout the tournament domain, stop and restore the boundary before proceeding.

### 5. Replace demo persistence and mount routes

- Replace the in-memory/browser tournament repository with a production repository at its composition root; do not change tournament services.
- Apply only `tournament_`-prefixed migrations or collections.
- Do not add tournament-owned columns to host surgeon or case tables.
- Mount the tournament routes beneath the host API prefix if necessary while preserving their payload contracts.
- Preserve host authentication and image-delivery policy.
- Make the profile score component call the mounted surgeon-score endpoint with the host surgeon ID.

### 6. Run provider conformance checks

Run the same contract suite against both fixture and host providers. At minimum, verify:

1. every returned entity has a non-empty string ID;
2. every eligible case references a surgeon the provider can resolve;
3. before and after URLs are present and authorized for display;
4. all returned cases match the requested procedure;
5. ineligible, unpublished, or unconsented cases are excluded;
6. repeated reads preserve IDs; and
7. missing entities return `null`, not fabricated placeholders.

Then exercise one complete path:

```text
request matchup
  -> confirm no surgeon identity is disclosed
  -> vote once
  -> replay with the same idempotency key
  -> confirm only one vote exists
  -> confirm both case ratings changed once
  -> confirm the winning case's surgeon score is recalculated
  -> render the score on that surgeon's profile
```

## API compatibility checklist

- `GET /api/tournament/matchup` returns two displayable, eligible cases and no surgeon identity.
- `POST /api/tournament/vote` accepts only a case in the issued matchup.
- `GET /api/tournament/surgeons/:id/score` accepts the host's surgeon ID unchanged.
- Optional leaderboard links use host-provided profile URLs rather than constructing host routes inside tournament logic.
- Signed or expiring image URLs are resolved at read time and are never persisted in tournament records.
- Error responses retain the documented status codes and stable machine-readable codes.

## Data and privacy checklist

- Confirm patient consent explicitly covers tournament comparison.
- Confirm cases and images are de-identified before the provider marks them eligible.
- Confirm logs and analytics do not capture image URLs or patient metadata.
- Confirm score UI includes the community-preference and medical-advice disclaimer.
- Confirm score visibility obeys the provisional sample threshold.
- Confirm rate limits and anonymous-session handling meet the host's abuse policy.

## Rollback

If the host adapter is not ready or a host schema change breaks it, switch registration back to the fixture provider for local development and disable tournament routes in shared environments. Because the tournament never mutates host data, rollback must not require reverting host surgeon, case, or image records.

Do not discard accepted tournament votes merely because a provider is temporarily unavailable. Restore provider access, validate IDs, and rebuild derived surgeon scores if needed.

## Merge acceptance

The merge is complete when:

- the host adapter passes the provider contract suite;
- only the composition-root provider registration differs from standalone development;
- no host surgeon/case/image data is duplicated into tournament tables;
- a real eligible host case can complete the matchup-to-profile-score path;
- an ineligible host case cannot enter a matchup;
- anonymization, idempotency, score provenance, consent, and disclaimer checks pass; and
- the host developer can change their internal schema by updating only the adapter.
