import type { CaseId, CaseRating } from "../domain/models";
import type { RatingEngine, RatingUpdate, RatingUpdateInput } from "../domain/ports";
import { DEFAULT_ELO_K_FACTOR, DEFAULT_ELO_RATING } from "./constants";

export interface EloRatingEngineOptions {
  readonly initialRating?: number;
  readonly kFactor?: number;
}

export class EloRatingEngine implements RatingEngine {
  readonly #initialRating: number;
  readonly #kFactor: number;

  constructor({
    initialRating = DEFAULT_ELO_RATING,
    kFactor = DEFAULT_ELO_K_FACTOR,
  }: EloRatingEngineOptions = {}) {
    if (!Number.isFinite(initialRating) || !Number.isFinite(kFactor) || kFactor <= 0) {
      throw new Error("Elo initialRating and kFactor must be finite; kFactor must be positive.");
    }
    this.#initialRating = initialRating;
    this.#kFactor = kFactor;
  }

  initialRating(caseId: CaseId): CaseRating {
    return {
      caseId,
      rating: this.#initialRating,
      comparisons: 0,
      wins: 0,
      losses: 0,
      updatedAt: null,
    };
  }

  update({ winner, loser, occurredAt }: RatingUpdateInput): RatingUpdate {
    if (winner.caseId === loser.caseId) {
      throw new Error("Winner and loser must be distinct cases.");
    }

    const expectedWinner = 1 / (1 + 10 ** ((loser.rating - winner.rating) / 400));
    const delta = this.#kFactor * (1 - expectedWinner);

    return {
      winner: {
        ...winner,
        rating: winner.rating + delta,
        comparisons: winner.comparisons + 1,
        wins: winner.wins + 1,
        updatedAt: occurredAt,
      },
      loser: {
        ...loser,
        rating: loser.rating - delta,
        comparisons: loser.comparisons + 1,
        losses: loser.losses + 1,
        updatedAt: occurredAt,
      },
    };
  }
}
