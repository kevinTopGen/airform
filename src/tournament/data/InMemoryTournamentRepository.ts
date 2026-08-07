import type {
  AnonymousVoterId,
  CaseId,
  CaseRating,
  MatchupId,
  MatchupRecord,
  PairKey,
  VoteRecord,
} from "../domain/models";
import type { TournamentRepository, VoteCommit } from "../domain/ports";
import { VoteRejectedError } from "../domain/errors";

export class InMemoryTournamentRepository implements TournamentRepository {
  readonly #matchups = new Map<MatchupId, MatchupRecord>();
  readonly #votes = new Map<MatchupId, VoteRecord>();
  readonly #ratings = new Map<CaseId, CaseRating>();
  readonly #votesByIdempotencyKey = new Map<string, VoteRecord>();

  async saveMatchup(matchup: MatchupRecord): Promise<void> {
    if (this.#matchups.has(matchup.id)) {
      throw new Error(`Matchup already exists: ${matchup.id}`);
    }
    this.#matchups.set(matchup.id, { ...matchup });
  }

  async getMatchup(matchupId: MatchupId): Promise<MatchupRecord | null> {
    return this.#matchups.get(matchupId) ?? null;
  }

  async listSeenPairKeys(voterId: AnonymousVoterId): Promise<ReadonlySet<PairKey>> {
    return new Set(
      [...this.#matchups.values()]
        .filter((matchup) => matchup.voterId === voterId)
        .map((matchup) => matchup.pairKey),
    );
  }

  async getVoteForMatchup(matchupId: MatchupId): Promise<VoteRecord | null> {
    return this.#votes.get(matchupId) ?? null;
  }

  async getVoteByIdempotencyKey(idempotencyKey: string): Promise<VoteRecord | null> {
    return this.#votesByIdempotencyKey.get(idempotencyKey) ?? null;
  }

  async getCaseRating(caseId: CaseId): Promise<CaseRating | null> {
    return this.#ratings.get(caseId) ?? null;
  }

  async listCaseRatings(): Promise<readonly CaseRating[]> {
    return [...this.#ratings.values()];
  }

  async commitVote({ vote, winnerRating, loserRating }: VoteCommit): Promise<void> {
    const matchup = this.#matchups.get(vote.matchupId);
    if (!matchup) {
      throw new VoteRejectedError("Cannot vote on a missing matchup.", "MATCHUP_NOT_FOUND");
    }
    if (this.#votes.has(vote.matchupId) || matchup.status === "voted") {
      throw new VoteRejectedError("This matchup has already been voted on.", "DUPLICATE_VOTE");
    }
    if (this.#votesByIdempotencyKey.has(vote.idempotencyKey)) {
      throw new VoteRejectedError(
        "This idempotency key has already been used.",
        "DUPLICATE_IDEMPOTENCY_KEY",
      );
    }

    this.#votes.set(vote.matchupId, { ...vote });
    this.#votesByIdempotencyKey.set(vote.idempotencyKey, { ...vote });
    this.#ratings.set(winnerRating.caseId, { ...winnerRating });
    this.#ratings.set(loserRating.caseId, { ...loserRating });
    this.#matchups.set(matchup.id, {
      ...matchup,
      status: "voted",
      votedAt: vote.createdAt,
    });
  }
}
