import type {
  AnonymousVoterId,
  CaseId,
  CaseRating,
  MatchupId,
  MatchupRecord,
  PairKey,
  Surgeon,
  SurgeonCase,
  VoteRecord,
} from "./models";

export interface CaseQuery {
  readonly procedureId?: string;
  readonly view?: string;
  readonly activeOnly?: boolean;
}

/** Integration seam for the other developer's authoritative case database. */
export interface SurgeonCaseProvider {
  getSurgeon(surgeonId: string): Promise<Surgeon | null>;
  listSurgeons(): Promise<readonly Surgeon[]>;
  getCase(caseId: CaseId): Promise<SurgeonCase | null>;
  listEligibleCases(query?: CaseQuery): Promise<readonly SurgeonCase[]>;
}

export interface MatchmakerInput {
  readonly cases: readonly SurgeonCase[];
  readonly ratings: ReadonlyMap<CaseId, CaseRating>;
  readonly excludedPairKeys: ReadonlySet<PairKey>;
}

export interface CasePair {
  readonly caseA: SurgeonCase;
  readonly caseB: SurgeonCase;
}

export interface Matchmaker {
  selectPair(input: MatchmakerInput): CasePair | null;
}

export interface RatingUpdateInput {
  readonly winner: CaseRating;
  readonly loser: CaseRating;
  readonly occurredAt: string;
}

export interface RatingUpdate {
  readonly winner: CaseRating;
  readonly loser: CaseRating;
}

export interface RatingEngine {
  initialRating(caseId: CaseId): CaseRating;
  update(input: RatingUpdateInput): RatingUpdate;
}

export interface VoteCommit {
  readonly vote: VoteRecord;
  readonly winnerRating: CaseRating;
  readonly loserRating: CaseRating;
}

/**
 * Persistence adapters must make commitVote atomic and reject a second vote for
 * the same matchup. A database implementation should enforce this with a
 * transaction plus a unique index on votes.matchup_id.
 */
export interface TournamentRepository {
  saveMatchup(matchup: MatchupRecord): Promise<void>;
  getMatchup(matchupId: MatchupId): Promise<MatchupRecord | null>;
  listSeenPairKeys(voterId: AnonymousVoterId): Promise<ReadonlySet<PairKey>>;
  getVoteForMatchup(matchupId: MatchupId): Promise<VoteRecord | null>;
  getVoteByIdempotencyKey(idempotencyKey: string): Promise<VoteRecord | null>;
  getCaseRating(caseId: CaseId): Promise<CaseRating | null>;
  listCaseRatings(): Promise<readonly CaseRating[]>;
  commitVote(commit: VoteCommit): Promise<void>;
}

export interface IdFactory {
  (): string;
}

export interface Clock {
  (): Date;
}
