import { beforeEach, describe, expect, it } from "vitest";
import { InMemoryTournamentRepository } from "../data/InMemoryTournamentRepository";
import { LocalFixtureCaseProvider } from "../data/LocalFixtureCaseProvider";
import { VoteRejectedError } from "../domain/errors";
import { createPairKey } from "../domain/models";
import { BalancedMatchmaker } from "../matchmaking/BalancedMatchmaker";
import { EloRatingEngine } from "../ratings/EloRatingEngine";
import { TournamentService } from "../services/TournamentService";
import { makeCase } from "./helpers";

describe("TournamentService", () => {
  const cases = [
    makeCase("a", "surgeon-a"),
    makeCase("b", "surgeon-b"),
    makeCase("c", "surgeon-c"),
  ];
  let repository: InMemoryTournamentRepository;
  let sequence: number;
  let service: TournamentService;

  beforeEach(() => {
    repository = new InMemoryTournamentRepository();
    sequence = 0;
    service = new TournamentService({
      caseProvider: new LocalFixtureCaseProvider(cases),
      repository,
      matchmaker: new BalancedMatchmaker({ random: () => 0 }),
      ratingEngine: new EloRatingEngine(),
      idFactory: () => `id-${(sequence += 1)}`,
      clock: () => new Date("2026-08-06T12:00:00.000Z"),
    });
  });

  it("serves an anonymized matchup and stores its pair as seen", async () => {
    const matchup = await service.getNextMatchup({ voterId: "anon-1" });
    const seen = await repository.listSeenPairKeys("anon-1");

    expect(matchup.caseA).not.toHaveProperty("surgeonId");
    expect(matchup.caseB).not.toHaveProperty("surgeonId");
    expect(seen.has(createPairKey(matchup.caseA.id, matchup.caseB.id))).toBe(true);
  });

  it("does not repeat a served pair for the same voter", async () => {
    const first = await service.getNextMatchup({ voterId: "anon-1" });
    const second = await service.getNextMatchup({ voterId: "anon-1" });

    expect(createPairKey(first.caseA.id, first.caseB.id)).not.toBe(
      createPairKey(second.caseA.id, second.caseB.id),
    );
  });

  it("allows the same pair for a different anonymous voter", async () => {
    const first = await service.getNextMatchup({ voterId: "anon-1" });
    const second = await service.getNextMatchup({ voterId: "anon-2" });

    expect(createPairKey(first.caseA.id, first.caseB.id)).toBe(
      createPairKey(second.caseA.id, second.caseB.id),
    );
  });

  it("records only a selected member of the served pair and updates ratings", async () => {
    const matchup = await service.getNextMatchup({ voterId: "anon-1" });
    const result = await service.submitVote({
      matchupId: matchup.id,
      voterId: "anon-1",
      selectedCaseId: matchup.caseB.id,
      idempotencyKey: "vote-1",
    });

    expect(result.winnerRating).toBe(1516);
    expect(result.loserRating).toBe(1484);
    expect((await repository.getCaseRating(matchup.caseB.id))?.wins).toBe(1);
  });

  it("rejects pair tampering, voter mismatch, and a duplicate vote", async () => {
    const matchup = await service.getNextMatchup({ voterId: "anon-1" });

    await expect(
      service.submitVote({
        matchupId: matchup.id,
        voterId: "anon-1",
        selectedCaseId: "c",
        idempotencyKey: "tampered",
      }),
    ).rejects.toMatchObject({ code: "INVALID_SELECTION" });
    await expect(
      service.submitVote({
        matchupId: matchup.id,
        voterId: "anon-other",
        selectedCaseId: matchup.caseA.id,
        idempotencyKey: "wrong-voter",
      }),
    ).rejects.toMatchObject({ code: "VOTER_MISMATCH" });

    await service.submitVote({
      matchupId: matchup.id,
      voterId: "anon-1",
      selectedCaseId: matchup.caseA.id,
      idempotencyKey: "accepted",
    });
    await expect(
      service.submitVote({
        matchupId: matchup.id,
        voterId: "anon-1",
        selectedCaseId: matchup.caseA.id,
        idempotencyKey: "another-key",
      }),
    ).rejects.toBeInstanceOf(VoteRejectedError);
  });

  it("aggregates profile-ready surgeon scores after voting", async () => {
    const matchup = await service.getNextMatchup({ voterId: "anon-1" });
    await service.submitVote({
      matchupId: matchup.id,
      voterId: "anon-1",
      selectedCaseId: matchup.caseA.id,
      idempotencyKey: "score-vote",
    });
    const scores = await service.getSurgeonScores("rhinoplasty");

    expect(scores).toHaveLength(3);
    expect(scores[0].surgeonId).toBe(cases.find((item) => item.id === matchup.caseA.id)?.surgeonId);
    expect(scores[0].rating).toBeGreaterThan(1500);
    expect(scores[0].score).toBeNull();
    expect(scores[0].provisional).toBe(true);

    const profileScore = await service.getSurgeonScore(scores[0].surgeonId, "rhinoplasty");
    expect(profileScore).toEqual(scores[0]);
  });

  it("returns the original vote for an exact idempotent replay", async () => {
    const matchup = await service.getNextMatchup({ voterId: "anon-1" });
    const request = {
      matchupId: matchup.id,
      voterId: "anon-1",
      selectedCaseId: matchup.caseA.id,
      idempotencyKey: "stable-request-key",
    };

    const first = await service.submitVote(request);
    const replay = await service.submitVote(request);

    expect(replay.voteId).toBe(first.voteId);
    expect((await repository.getCaseRating(matchup.caseA.id))?.comparisons).toBe(1);
  });

  it("rejects an expired matchup", async () => {
    let now = new Date("2026-08-06T12:00:00.000Z");
    const expiringService = new TournamentService({
      caseProvider: new LocalFixtureCaseProvider(cases),
      repository,
      matchmaker: new BalancedMatchmaker({ random: () => 0 }),
      ratingEngine: new EloRatingEngine(),
      idFactory: () => `expiry-${(sequence += 1)}`,
      clock: () => now,
      matchupTtlMs: 1000,
    });
    const matchup = await expiringService.getNextMatchup({ voterId: "anon-1" });
    now = new Date("2026-08-06T12:00:02.000Z");

    await expect(
      expiringService.submitVote({
        matchupId: matchup.id,
        voterId: "anon-1",
        selectedCaseId: matchup.caseA.id,
        idempotencyKey: "expired-vote",
      }),
    ).rejects.toMatchObject({ code: "MATCHUP_EXPIRED" });
  });
});
