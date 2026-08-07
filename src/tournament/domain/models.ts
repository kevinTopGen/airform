export type SurgeonId = string;
export type CaseId = string;
export type MatchupId = string;
export type VoteId = string;
export type AnonymousVoterId = string;
export type PairKey = string;

export type CaseImagePhase = "before" | "after";
export type CaseStatus = "active" | "inactive";

export interface Surgeon {
  readonly id: SurgeonId;
  readonly name: string;
  readonly slug?: string;
  readonly profileImageUrl?: string;
}

export interface CaseImage {
  readonly id: string;
  readonly url: string;
  readonly phase: CaseImagePhase;
  readonly alt: string;
  readonly width?: number;
  readonly height?: number;
}

/**
 * Canonical case shape consumed by the tournament. The eventual host app only
 * needs an adapter that maps its own surgeon/case records into this interface.
 */
export interface SurgeonCase {
  readonly id: CaseId;
  readonly surgeonId: SurgeonId;
  readonly procedureId: string;
  readonly view: string;
  readonly status: CaseStatus;
  readonly images: {
    readonly before: CaseImage;
    readonly after: CaseImage;
  };
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Safe to send to a browser: deliberately excludes surgeonId and metadata. */
export interface AnonymousCase {
  readonly id: CaseId;
  readonly procedureId: string;
  readonly view: string;
  readonly images: SurgeonCase["images"];
}

export type MatchupStatus = "open" | "voted" | "expired";

export interface MatchupRecord {
  readonly id: MatchupId;
  readonly voterId: AnonymousVoterId;
  readonly caseAId: CaseId;
  readonly caseBId: CaseId;
  readonly pairKey: PairKey;
  readonly status: MatchupStatus;
  readonly createdAt: string;
  readonly expiresAt: string;
  readonly votedAt?: string;
}

export interface AnonymousMatchup {
  readonly id: MatchupId;
  readonly caseA: AnonymousCase;
  readonly caseB: AnonymousCase;
  readonly createdAt: string;
  readonly expiresAt: string;
}

export interface VoteRecord {
  readonly id: VoteId;
  readonly matchupId: MatchupId;
  readonly voterId: AnonymousVoterId;
  readonly winnerCaseId: CaseId;
  readonly loserCaseId: CaseId;
  readonly idempotencyKey: string;
  readonly createdAt: string;
}

export interface CaseRating {
  readonly caseId: CaseId;
  readonly rating: number;
  readonly comparisons: number;
  readonly wins: number;
  readonly losses: number;
  readonly updatedAt: string | null;
}

export interface SurgeonScore {
  readonly surgeonId: SurgeonId;
  /** Elo-like value after shrinkage toward the configured population prior. */
  readonly rating: number;
  /** UI-oriented 0-100 representation; hidden until evidence thresholds pass. */
  readonly score: number | null;
  readonly comparisons: number;
  readonly ratedCases: number;
  readonly totalCases: number;
  /** 0-1 evidence indicator; it is not a statistical confidence interval. */
  readonly confidence: number;
  readonly provisional: boolean;
  readonly algorithm: "elo-shrunk-global-mean-v1";
  readonly updatedAt: string | null;
}

export const toAnonymousCase = (surgeonCase: SurgeonCase): AnonymousCase => ({
  id: surgeonCase.id,
  procedureId: surgeonCase.procedureId,
  view: surgeonCase.view,
  images: surgeonCase.images,
});

export const createPairKey = (first: CaseId, second: CaseId): PairKey => {
  if (first === second) {
    throw new Error("A matchup requires two distinct cases.");
  }

  return JSON.stringify([first, second].sort());
};
