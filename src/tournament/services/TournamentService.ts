import {
  createPairKey,
  toAnonymousCase,
  type AnonymousMatchup,
  type AnonymousVoterId,
  type CaseRating,
  type SurgeonScore,
} from "../domain/models";
import {
  MatchupNotFoundError,
  NoMatchupAvailableError,
  VoteRejectedError,
} from "../domain/errors";
import type {
  Clock,
  IdFactory,
  Matchmaker,
  RatingEngine,
  SurgeonCaseProvider,
  TournamentRepository,
} from "../domain/ports";
import { calculateSurgeonScores, type SurgeonScoreOptions } from "../ratings/surgeonScores";

export interface TournamentServiceDependencies {
  readonly caseProvider: SurgeonCaseProvider;
  readonly repository: TournamentRepository;
  readonly matchmaker: Matchmaker;
  readonly ratingEngine: RatingEngine;
  readonly idFactory?: IdFactory;
  readonly clock?: Clock;
  readonly surgeonScoreOptions?: SurgeonScoreOptions;
  readonly matchupTtlMs?: number;
}

export interface NextMatchupRequest {
  readonly voterId: AnonymousVoterId;
  readonly procedureId?: string;
  readonly view?: string;
}

export interface SubmitVoteRequest {
  readonly matchupId: string;
  readonly voterId: AnonymousVoterId;
  readonly selectedCaseId: string;
  readonly idempotencyKey: string;
}

export interface VoteResult {
  readonly voteId: string;
  readonly matchupId: string;
  readonly selectedCaseId: string;
  readonly winnerRating: number;
  readonly loserRating: number;
}

const defaultIdFactory = (): string => {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
};

const defaultClock = (): Date => new Date();

const requireVoterId = (voterId: string): void => {
  if (!voterId.trim()) {
    throw new VoteRejectedError("An anonymous voter id is required.", "INVALID_VOTER");
  }
};

const caseQuery = (procedureId?: string, view?: string) => ({
  activeOnly: true as const,
  ...(procedureId ? { procedureId } : {}),
  ...(view ? { view } : {}),
});

export class TournamentService {
  readonly #caseProvider: SurgeonCaseProvider;
  readonly #repository: TournamentRepository;
  readonly #matchmaker: Matchmaker;
  readonly #ratingEngine: RatingEngine;
  readonly #idFactory: IdFactory;
  readonly #clock: Clock;
  readonly #surgeonScoreOptions: SurgeonScoreOptions;
  readonly #matchupTtlMs: number;

  constructor({
    caseProvider,
    repository,
    matchmaker,
    ratingEngine,
    idFactory = defaultIdFactory,
    clock = defaultClock,
    surgeonScoreOptions = {},
    matchupTtlMs = 5 * 60 * 1000,
  }: TournamentServiceDependencies) {
    this.#caseProvider = caseProvider;
    this.#repository = repository;
    this.#matchmaker = matchmaker;
    this.#ratingEngine = ratingEngine;
    this.#idFactory = idFactory;
    this.#clock = clock;
    this.#surgeonScoreOptions = surgeonScoreOptions;
    this.#matchupTtlMs = matchupTtlMs;
  }

  async getNextMatchup({ voterId, procedureId, view }: NextMatchupRequest): Promise<AnonymousMatchup> {
    requireVoterId(voterId);
    const [cases, ratings, excludedPairKeys] = await Promise.all([
      this.#caseProvider.listEligibleCases(caseQuery(procedureId, view)),
      this.#repository.listCaseRatings(),
      this.#repository.listSeenPairKeys(voterId),
    ]);
    const ratingsByCase = new Map(ratings.map((rating) => [rating.caseId, rating]));
    const pair = this.#matchmaker.selectPair({ cases, ratings: ratingsByCase, excludedPairKeys });
    if (!pair) {
      throw new NoMatchupAvailableError();
    }

    const id = this.#idFactory();
    const now = this.#clock();
    const createdAt = now.toISOString();
    const expiresAt = new Date(now.getTime() + this.#matchupTtlMs).toISOString();
    await this.#repository.saveMatchup({
      id,
      voterId,
      caseAId: pair.caseA.id,
      caseBId: pair.caseB.id,
      pairKey: createPairKey(pair.caseA.id, pair.caseB.id),
      status: "open",
      createdAt,
      expiresAt,
    });

    return {
      id,
      caseA: toAnonymousCase(pair.caseA),
      caseB: toAnonymousCase(pair.caseB),
      createdAt,
      expiresAt,
    };
  }

  async submitVote({
    matchupId,
    voterId,
    selectedCaseId,
    idempotencyKey,
  }: SubmitVoteRequest): Promise<VoteResult> {
    requireVoterId(voterId);
    if (!idempotencyKey.trim()) {
      throw new VoteRejectedError("An idempotency key is required.", "INVALID_IDEMPOTENCY_KEY");
    }
    const replayedVote = await this.#repository.getVoteByIdempotencyKey(idempotencyKey);
    if (replayedVote) {
      if (
        replayedVote.matchupId !== matchupId ||
        replayedVote.voterId !== voterId ||
        replayedVote.winnerCaseId !== selectedCaseId
      ) {
        throw new VoteRejectedError(
          "This idempotency key was used for a different vote.",
          "IDEMPOTENCY_CONFLICT",
        );
      }
      const [winnerRating, loserRating] = await Promise.all([
        this.#repository.getCaseRating(replayedVote.winnerCaseId),
        this.#repository.getCaseRating(replayedVote.loserCaseId),
      ]);
      return {
        voteId: replayedVote.id,
        matchupId,
        selectedCaseId,
        winnerRating: winnerRating?.rating ?? NaN,
        loserRating: loserRating?.rating ?? NaN,
      };
    }
    const matchup = await this.#repository.getMatchup(matchupId);
    if (!matchup) {
      throw new MatchupNotFoundError();
    }
    if (matchup.voterId !== voterId) {
      throw new VoteRejectedError("This matchup belongs to a different voter.", "VOTER_MISMATCH");
    }
    if (await this.#repository.getVoteForMatchup(matchupId)) {
      throw new VoteRejectedError("This matchup has already been voted on.", "DUPLICATE_VOTE");
    }
    if (this.#clock().getTime() >= new Date(matchup.expiresAt).getTime()) {
      throw new VoteRejectedError("This matchup has expired.", "MATCHUP_EXPIRED");
    }
    if (selectedCaseId !== matchup.caseAId && selectedCaseId !== matchup.caseBId) {
      throw new VoteRejectedError("Selected case is not part of this matchup.", "INVALID_SELECTION");
    }

    const loserCaseId = selectedCaseId === matchup.caseAId ? matchup.caseBId : matchup.caseAId;
    const [winnerCase, loserCase, storedWinnerRating, storedLoserRating] = await Promise.all([
      this.#caseProvider.getCase(selectedCaseId),
      this.#caseProvider.getCase(loserCaseId),
      this.#repository.getCaseRating(selectedCaseId),
      this.#repository.getCaseRating(loserCaseId),
    ]);
    if (!winnerCase || !loserCase) {
      throw new VoteRejectedError("A case in this matchup is no longer available.", "CASE_NOT_FOUND");
    }

    const occurredAt = this.#clock().toISOString();
    const update = this.#ratingEngine.update({
      winner: storedWinnerRating ?? this.#ratingEngine.initialRating(selectedCaseId),
      loser: storedLoserRating ?? this.#ratingEngine.initialRating(loserCaseId),
      occurredAt,
    });
    const voteId = this.#idFactory();
    await this.#repository.commitVote({
      vote: {
        id: voteId,
        matchupId,
        voterId,
        winnerCaseId: selectedCaseId,
        loserCaseId,
        idempotencyKey,
        createdAt: occurredAt,
      },
      winnerRating: update.winner,
      loserRating: update.loser,
    });

    return {
      voteId,
      matchupId,
      selectedCaseId,
      winnerRating: update.winner.rating,
      loserRating: update.loser.rating,
    };
  }

  async getSurgeonScores(procedureId?: string): Promise<readonly SurgeonScore[]> {
    const [cases, ratings] = await Promise.all([
      this.#caseProvider.listEligibleCases(caseQuery(procedureId)),
      this.#repository.listCaseRatings(),
    ]);
    return calculateSurgeonScores(cases, ratings, this.#surgeonScoreOptions);
  }

  async getSurgeonScore(
    surgeonId: string,
    procedureId?: string,
  ): Promise<SurgeonScore | null> {
    if (!(await this.#caseProvider.getSurgeon(surgeonId))) {
      return null;
    }
    const scores = await this.getSurgeonScores(procedureId);
    return scores.find((score) => score.surgeonId === surgeonId) ?? null;
  }

  async getLeaderboard(
    procedureId?: string,
    limit = 25,
  ): Promise<readonly SurgeonScore[]> {
    const cappedLimit = Math.max(0, Math.min(100, Math.floor(limit)));
    return (await this.getSurgeonScores(procedureId))
      .filter((score) => !score.provisional)
      .slice(0, cappedLimit);
  }
}
