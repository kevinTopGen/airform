import type { CaseRating, SurgeonCase, SurgeonScore } from "../domain/models";
import {
  DEFAULT_COMMUNITY_SCORE_CENTER,
  DEFAULT_ELO_RATING,
  DEFAULT_MIN_RATED_CASES,
  DEFAULT_MIN_SURGEON_COMPARISONS,
  DEFAULT_RATING_POINTS_PER_COMMUNITY_POINT,
  DEFAULT_SURGEON_PRIOR_COMPARISONS,
} from "./constants";

export interface SurgeonScoreOptions {
  readonly priorRating?: number;
  readonly priorComparisons?: number;
  readonly displayPointsPerRating?: number;
  readonly minimumRatedCases?: number;
  readonly minimumComparisons?: number;
}

const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(maximum, Math.max(minimum, value));

export const calculateGlobalRatingMean = (
  ratings: readonly CaseRating[],
  fallback = DEFAULT_ELO_RATING,
): number => {
  const rated = ratings.filter((rating) => rating.comparisons > 0);
  const comparisons = rated.reduce((total, rating) => total + rating.comparisons, 0);
  if (comparisons === 0) {
    return fallback;
  }
  return (
    rated.reduce((total, rating) => total + rating.rating * rating.comparisons, 0) / comparisons
  );
};

export const normalizeCommunityScore = (
  rating: number,
  centerRating: number,
  ratingPointsPerCommunityPoint = DEFAULT_RATING_POINTS_PER_COMMUNITY_POINT,
): number => {
  if (ratingPointsPerCommunityPoint <= 0) {
    throw new Error("Community score scale must be positive.");
  }
  return clamp(
    DEFAULT_COMMUNITY_SCORE_CENTER +
      (rating - centerRating) / ratingPointsPerCommunityPoint,
    0,
    100,
  );
};

export const calculateSurgeonScores = (
  cases: readonly SurgeonCase[],
  ratings: readonly CaseRating[],
  {
    priorRating,
    priorComparisons = DEFAULT_SURGEON_PRIOR_COMPARISONS,
    displayPointsPerRating = DEFAULT_RATING_POINTS_PER_COMMUNITY_POINT,
    minimumRatedCases = DEFAULT_MIN_RATED_CASES,
    minimumComparisons = DEFAULT_MIN_SURGEON_COMPARISONS,
  }: SurgeonScoreOptions = {},
): readonly SurgeonScore[] => {
  if (
    priorComparisons < 0 ||
    displayPointsPerRating <= 0 ||
    minimumRatedCases < 0 ||
    minimumComparisons < 0
  ) {
    throw new Error("Score prior must be non-negative and display scale must be positive.");
  }

  const ratingsByCase = new Map(ratings.map((rating) => [rating.caseId, rating]));
  const populationRating = priorRating ?? calculateGlobalRatingMean(ratings);
  const casesBySurgeon = new Map<string, SurgeonCase[]>();

  for (const surgeonCase of cases) {
    const existing = casesBySurgeon.get(surgeonCase.surgeonId) ?? [];
    existing.push(surgeonCase);
    casesBySurgeon.set(surgeonCase.surgeonId, existing);
  }

  return [...casesBySurgeon.entries()]
    .map(([surgeonId, surgeonCases]): SurgeonScore => {
      const caseRatings = surgeonCases
        .map((surgeonCase) => ratingsByCase.get(surgeonCase.id))
        .filter((rating): rating is CaseRating => Boolean(rating && rating.comparisons > 0));
      const comparisons = caseRatings.reduce((total, rating) => total + rating.comparisons, 0);
      const weightedRatingTotal = caseRatings.reduce(
        (total, rating) => total + rating.rating * rating.comparisons,
        0,
      );
      const denominator = comparisons + priorComparisons;
      const rating =
        denominator === 0
          ? populationRating
          : (weightedRatingTotal + populationRating * priorComparisons) / denominator;
      const provisional =
        caseRatings.length < minimumRatedCases || comparisons < minimumComparisons;
      const updatedAt = caseRatings
        .map((caseRating) => caseRating.updatedAt)
        .filter((value): value is string => Boolean(value))
        .sort()
        .at(-1) ?? null;

      return {
        surgeonId,
        rating,
        score: provisional
          ? null
          : normalizeCommunityScore(rating, populationRating, displayPointsPerRating),
        comparisons,
        ratedCases: caseRatings.length,
        totalCases: surgeonCases.length,
        confidence: denominator === 0 ? 0 : comparisons / denominator,
        provisional,
        algorithm: "elo-shrunk-global-mean-v1",
        updatedAt,
      };
    })
    .sort((left, right) => right.rating - left.rating || left.surgeonId.localeCompare(right.surgeonId));
};
