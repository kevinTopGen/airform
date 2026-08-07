import { beforeEach, describe, expect, it, vi } from "vitest";
import type { MatchupRecord } from "../domain/models";

const values = new Map<string, string>();

vi.stubGlobal("localStorage", {
  getItem: (key: string) => values.get(key) ?? null,
  setItem: (key: string, value: string) => values.set(key, value),
});

const { BrowserTournamentRepository } = await import("../ui/browserTournament");

const matchup = (
  id: string,
  createdAt: string,
  expiresAt = "2099-01-01T00:00:00.000Z",
): MatchupRecord => ({
  id,
  voterId: "voter-1",
  caseAId: `${id}-a`,
  caseBId: `${id}-b`,
  pairKey: `["${id}-a","${id}-b"]`,
  status: "open",
  createdAt,
  expiresAt,
});

describe("BrowserTournamentRepository open matchup lifecycle", () => {
  beforeEach(() => values.clear());

  it("resumes the latest open matchup instead of consuming another pair", async () => {
    const repository = new BrowserTournamentRepository();
    await repository.saveMatchup(matchup("first", "2026-08-06T12:00:00.000Z"));
    await repository.saveMatchup(matchup("second", "2026-08-06T12:01:00.000Z"));

    expect(repository.getLatestOpenMatchup("voter-1")?.id).toBe("second");

    repository.expireMatchup("second");
    expect(repository.getLatestOpenMatchup("voter-1")?.id).toBe("first");
  });

  it("does not resume expired matchups", async () => {
    const repository = new BrowserTournamentRepository();
    await repository.saveMatchup(
      matchup("expired", "2020-01-01T00:00:00.000Z", "2020-01-01T00:05:00.000Z"),
    );

    expect(repository.getLatestOpenMatchup("voter-1")).toBeNull();
  });
});
