import { describe, expect, it } from "vitest";
import { createPairKey } from "../domain/models";
import { BalancedMatchmaker } from "../matchmaking/BalancedMatchmaker";
import { makeCase, makeRating } from "./helpers";

describe("BalancedMatchmaker", () => {
  const cases = [
    makeCase("a1", "surgeon-a"),
    makeCase("a2", "surgeon-a"),
    makeCase("b1", "surgeon-b"),
    makeCase("c1", "surgeon-c"),
  ];

  it("never compares cases from the same surgeon", () => {
    const matchmaker = new BalancedMatchmaker({ random: () => 0 });
    const selected = matchmaker.selectPair({
      cases: cases.slice(0, 3),
      ratings: new Map(),
      excludedPairKeys: new Set([createPairKey("a1", "b1")]),
    });

    expect(selected?.caseA.id).toBe("a2");
    expect(selected?.caseB.id).toBe("b1");
  });

  it("filters already-served pairs and returns null after exhaustion", () => {
    const matchmaker = new BalancedMatchmaker({ random: () => 0 });
    const selected = matchmaker.selectPair({
      cases: [cases[0], cases[2]],
      ratings: new Map(),
      excludedPairKeys: new Set([createPairKey("a1", "b1")]),
    });

    expect(selected).toBeNull();
  });

  it("prefers the least exposed pair, then the closest rating", () => {
    const matchmaker = new BalancedMatchmaker({ random: () => 0 });
    const selected = matchmaker.selectPair({
      cases,
      ratings: new Map([
        ["a1", makeRating("a1", 1500, 2)],
        ["a2", makeRating("a2", 1500, 8)],
        ["b1", makeRating("b1", 1505, 2)],
        ["c1", makeRating("c1", 1700, 2)],
      ]),
      excludedPairKeys: new Set(),
    });

    expect([selected?.caseA.id, selected?.caseB.id]).toEqual(["a1", "b1"]);
  });

  it("only pairs equivalent procedure views", () => {
    const matchmaker = new BalancedMatchmaker({ random: () => 0 });
    const selected = matchmaker.selectPair({
      cases: [makeCase("front", "one", "front"), makeCase("profile", "two", "profile")],
      ratings: new Map(),
      excludedPairKeys: new Set(),
    });

    expect(selected).toBeNull();
  });
});
