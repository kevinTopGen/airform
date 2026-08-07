import { describe, expect, it } from "vitest";
import { EloRatingEngine } from "../ratings/EloRatingEngine";

describe("EloRatingEngine", () => {
  it("starts cases at 1500 and applies a symmetric K=32 update", () => {
    const engine = new EloRatingEngine();
    const first = engine.initialRating("first");
    const second = engine.initialRating("second");
    const update = engine.update({
      winner: first,
      loser: second,
      occurredAt: "2026-08-06T12:00:00.000Z",
    });

    expect(first.rating).toBe(1500);
    expect(update.winner.rating).toBe(1516);
    expect(update.loser.rating).toBe(1484);
    expect(update.winner.comparisons).toBe(1);
    expect(update.winner.wins).toBe(1);
    expect(update.loser.losses).toBe(1);
    expect(update.winner.rating + update.loser.rating).toBe(3000);
  });

  it("awards a larger change when an underdog wins", () => {
    const engine = new EloRatingEngine();
    const update = engine.update({
      winner: { ...engine.initialRating("underdog"), rating: 1300 },
      loser: { ...engine.initialRating("favorite"), rating: 1700 },
      occurredAt: "2026-08-06T12:00:00.000Z",
    });

    expect(update.winner.rating - 1300).toBeGreaterThan(29);
    expect(1700 - update.loser.rating).toBeGreaterThan(29);
  });

  it("rejects self-matches", () => {
    const engine = new EloRatingEngine();
    expect(() =>
      engine.update({
        winner: engine.initialRating("same"),
        loser: engine.initialRating("same"),
        occurredAt: "2026-08-06T12:00:00.000Z",
      }),
    ).toThrow("distinct");
  });
});
