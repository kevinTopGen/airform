import { createPairKey } from "../domain/models";
import type { CasePair, Matchmaker, MatchmakerInput } from "../domain/ports";
import { DEFAULT_ELO_RATING } from "../ratings/constants";

export interface BalancedMatchmakerOptions {
  readonly random?: () => number;
}

interface Candidate extends CasePair {
  readonly totalComparisons: number;
  readonly ratingGap: number;
  readonly tieBreaker: number;
}

/**
 * Prefers under-exposed, similarly-rated cases from different surgeons. Pair
 * history is supplied by persistence, making the policy stateless/replaceable.
 */
export class BalancedMatchmaker implements Matchmaker {
  readonly #random: () => number;

  constructor({ random = Math.random }: BalancedMatchmakerOptions = {}) {
    this.#random = random;
  }

  selectPair({ cases, ratings, excludedPairKeys }: MatchmakerInput): CasePair | null {
    const candidates: Candidate[] = [];

    for (const [firstIndex, caseA] of cases.entries()) {
      for (const caseB of cases.slice(firstIndex + 1)) {
        if (
          caseA.surgeonId === caseB.surgeonId ||
          caseA.procedureId !== caseB.procedureId ||
          caseA.view !== caseB.view ||
          excludedPairKeys.has(createPairKey(caseA.id, caseB.id))
        ) {
          continue;
        }

        const ratingA = ratings.get(caseA.id);
        const ratingB = ratings.get(caseB.id);
        candidates.push({
          caseA,
          caseB,
          totalComparisons: (ratingA?.comparisons ?? 0) + (ratingB?.comparisons ?? 0),
          ratingGap: Math.abs(
            (ratingA?.rating ?? DEFAULT_ELO_RATING) - (ratingB?.rating ?? DEFAULT_ELO_RATING),
          ),
          tieBreaker: this.#random(),
        });
      }
    }

    candidates.sort(
      (left, right) =>
        left.totalComparisons - right.totalComparisons ||
        left.ratingGap - right.ratingGap ||
        left.tieBreaker - right.tieBreaker,
    );

    const selected = candidates[0];
    return selected ? { caseA: selected.caseA, caseB: selected.caseB } : null;
  }
}
