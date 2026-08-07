import { VoteRejectedError } from "../domain/errors";
import type {
  AnonymousMatchup,
  AnonymousVoterId,
  CaseId,
  CaseRating,
  MatchupId,
  MatchupRecord,
  PairKey,
  VoteRecord,
} from "../domain/models";
import { toAnonymousCase } from "../domain/models";
import type { TournamentRepository, VoteCommit } from "../domain/ports";
import { localFixtureCases, localFixtureSurgeons } from "../data/fixtures";
import { LocalFixtureCaseProvider } from "../data/LocalFixtureCaseProvider";
import { BalancedMatchmaker } from "../matchmaking/BalancedMatchmaker";
import { EloRatingEngine } from "../ratings/EloRatingEngine";
import { TournamentService } from "../services/TournamentService";

const STORAGE_KEY = "tournament_state_v1";
const VOTER_KEY = "tournament_voter_id";

interface TournamentSnapshot {
  matchups: MatchupRecord[];
  votes: VoteRecord[];
  ratings: CaseRating[];
}

const emptySnapshot = (): TournamentSnapshot => ({ matchups: [], votes: [], ratings: [] });

const copy = <T,>(value: T): T => structuredClone(value);

/** Demo-only persistence adapter. The host merge replaces this class, not the UI. */
export class BrowserTournamentRepository implements TournamentRepository {
  #snapshot: TournamentSnapshot;

  constructor() {
    this.#snapshot = this.#read();
  }

  #read(): TournamentSnapshot {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return emptySnapshot();
      const parsed = JSON.parse(raw) as Partial<TournamentSnapshot>;
      return {
        matchups: Array.isArray(parsed.matchups) ? parsed.matchups : [],
        votes: Array.isArray(parsed.votes) ? parsed.votes : [],
        ratings: Array.isArray(parsed.ratings) ? parsed.ratings : [],
      };
    } catch {
      return emptySnapshot();
    }
  }

  #publish(next: TournamentSnapshot): void {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    this.#snapshot = next;
  }

  async saveMatchup(matchup: MatchupRecord): Promise<void> {
    if (this.#snapshot.matchups.some((item) => item.id === matchup.id)) {
      throw new Error(`Matchup already exists: ${matchup.id}`);
    }
    this.#publish({ ...this.#snapshot, matchups: [...this.#snapshot.matchups, copy(matchup)] });
  }

  async getMatchup(matchupId: MatchupId): Promise<MatchupRecord | null> {
    return copy(this.#snapshot.matchups.find((item) => item.id === matchupId) ?? null);
  }

  async listSeenPairKeys(voterId: AnonymousVoterId): Promise<ReadonlySet<PairKey>> {
    return new Set(
      this.#snapshot.matchups
        .filter((item) => item.voterId === voterId)
        .map((item) => item.pairKey),
    );
  }

  async getVoteForMatchup(matchupId: MatchupId): Promise<VoteRecord | null> {
    return copy(this.#snapshot.votes.find((item) => item.matchupId === matchupId) ?? null);
  }

  async getVoteByIdempotencyKey(idempotencyKey: string): Promise<VoteRecord | null> {
    return copy(this.#snapshot.votes.find((item) => item.idempotencyKey === idempotencyKey) ?? null);
  }

  async getCaseRating(caseId: CaseId): Promise<CaseRating | null> {
    return copy(this.#snapshot.ratings.find((item) => item.caseId === caseId) ?? null);
  }

  async listCaseRatings(): Promise<readonly CaseRating[]> {
    return copy(this.#snapshot.ratings);
  }

  async commitVote({ vote, winnerRating, loserRating }: VoteCommit): Promise<void> {
    const matchup = this.#snapshot.matchups.find((item) => item.id === vote.matchupId);
    if (!matchup) {
      throw new VoteRejectedError("Cannot vote on a missing matchup.", "MATCHUP_NOT_FOUND");
    }
    if (
      matchup.status === "voted" ||
      this.#snapshot.votes.some((item) => item.matchupId === vote.matchupId)
    ) {
      throw new VoteRejectedError("This matchup has already been voted on.", "DUPLICATE_VOTE");
    }
    if (this.#snapshot.votes.some((item) => item.idempotencyKey === vote.idempotencyKey)) {
      throw new VoteRejectedError(
        "This idempotency key has already been used.",
        "DUPLICATE_IDEMPOTENCY_KEY",
      );
    }

    const ratings = this.#snapshot.ratings.filter(
      (item) => item.caseId !== winnerRating.caseId && item.caseId !== loserRating.caseId,
    );
    const matchups = this.#snapshot.matchups.map((item) =>
      item.id === matchup.id ? { ...item, status: "voted" as const, votedAt: vote.createdAt } : item,
    );
    this.#publish({
      matchups,
      votes: [...this.#snapshot.votes, copy(vote)],
      ratings: [...ratings, copy(winnerRating), copy(loserRating)],
    });
  }

  getVoteCount(): number {
    return this.#snapshot.votes.length;
  }

  getVoterVoteCount(voterId: string): number {
    return this.#snapshot.votes.filter((item) => item.voterId === voterId).length;
  }

  getLatestOpenMatchup(voterId: AnonymousVoterId): MatchupRecord | null {
    const now = Date.now();
    const openMatchups = this.#snapshot.matchups.filter(
      (item) =>
        item.voterId === voterId &&
        item.status === "open" &&
        new Date(item.expiresAt).getTime() > now,
    );
    return copy(openMatchups.at(-1) ?? null);
  }

  expireMatchup(matchupId: MatchupId): void {
    const matchups = this.#snapshot.matchups.map((item) =>
      item.id === matchupId && item.status === "open"
        ? { ...item, status: "expired" as const }
        : item,
    );
    this.#publish({ ...this.#snapshot, matchups });
  }
}

export const browserRepository = new BrowserTournamentRepository();
export const browserCaseProvider = new LocalFixtureCaseProvider(
  localFixtureCases,
  localFixtureSurgeons,
);
export const tournamentService = new TournamentService({
  caseProvider: browserCaseProvider,
  repository: browserRepository,
  matchmaker: new BalancedMatchmaker(),
  ratingEngine: new EloRatingEngine(),
  // Demo fixtures contain one case per practice. Production keeps the stricter defaults.
  surgeonScoreOptions: { minimumRatedCases: 1, minimumComparisons: 1 },
});

export const resumeOpenMatchup = async (
  voterId: AnonymousVoterId,
): Promise<AnonymousMatchup | null> => {
  const record = browserRepository.getLatestOpenMatchup(voterId);
  if (!record) return null;

  const [caseA, caseB] = await Promise.all([
    browserCaseProvider.getCase(record.caseAId),
    browserCaseProvider.getCase(record.caseBId),
  ]);
  if (!caseA || !caseB) {
    browserRepository.expireMatchup(record.id);
    return null;
  }

  return {
    id: record.id,
    caseA: toAnonymousCase(caseA),
    caseB: toAnonymousCase(caseB),
    createdAt: record.createdAt,
    expiresAt: record.expiresAt,
  };
};

const randomId = (): string =>
  typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

export const getOrCreateVoterId = (): string => {
  const existing = localStorage.getItem(VOTER_KEY);
  if (existing) return existing;
  const voterId = `anonymous-${randomId()}`;
  localStorage.setItem(VOTER_KEY, voterId);
  return voterId;
};

export const beginNewAnonymousSession = (): string => {
  const voterId = `anonymous-${randomId()}`;
  localStorage.setItem(VOTER_KEY, voterId);
  return voterId;
};

export const createIdempotencyKey = (): string => `vote-${randomId()}`;
