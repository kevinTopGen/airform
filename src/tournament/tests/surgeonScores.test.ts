import { describe, expect, it } from "vitest";
import {
  calculateGlobalRatingMean,
  calculateSurgeonScores,
  normalizeCommunityScore,
} from "../ratings/surgeonScores";
import { makeCase, makeRating } from "./helpers";

describe("surgeon score aggregation", () => {
  it("shrinks sparse surgeon ratings toward the global mean", () => {
    const cases = [
      makeCase("a", "surgeon-a"),
      makeCase("b", "surgeon-b"),
      makeCase("c", "surgeon-c"),
    ];
    const ratings = [
      makeRating("a", 1700, 1),
      makeRating("b", 1400, 10),
      makeRating("c", 1500, 10),
    ];
    const globalMean = calculateGlobalRatingMean(ratings);
    const scores = calculateSurgeonScores(cases, ratings);
    const sparse = scores.find((score) => score.surgeonId === "surgeon-a");

    expect(globalMean).toBeCloseTo(1461.9, 2);
    expect(sparse?.rating).toBeGreaterThan(globalMean);
    expect(sparse?.rating).toBeLessThan(1500);
    expect(sparse?.confidence).toBeCloseTo(1 / 11);
  });

  it("returns an unrated surgeon at the population prior with zero confidence", () => {
    const scores = calculateSurgeonScores([makeCase("a", "surgeon-a")], []);

    expect(scores[0]).toMatchObject({
      surgeonId: "surgeon-a",
      rating: 1500,
      score: null,
      comparisons: 0,
      confidence: 0,
      provisional: true,
    });
  });

  it("normalizes and clamps a Community Score to 0-100", () => {
    expect(normalizeCommunityScore(1500, 1500)).toBe(50);
    expect(normalizeCommunityScore(2500, 1500)).toBe(100);
    expect(normalizeCommunityScore(500, 1500)).toBe(0);
  });

  it("publishes a score only after three cases and ten comparisons", () => {
    const cases = [
      makeCase("a1", "surgeon-a"),
      makeCase("a2", "surgeon-a"),
      makeCase("a3", "surgeon-a"),
    ];
    const ratings = [
      makeRating("a1", 1550, 4),
      makeRating("a2", 1540, 3),
      makeRating("a3", 1530, 3),
    ];
    const score = calculateSurgeonScores(cases, ratings, { priorRating: 1500 })[0];

    expect(score.provisional).toBe(false);
    expect(score.score).toBeGreaterThan(50);
    expect(score.algorithm).toBe("elo-shrunk-global-mean-v1");
  });
});
