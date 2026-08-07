export class TournamentError extends Error {
  constructor(
    message: string,
    readonly code: string,
  ) {
    super(message);
    this.name = "TournamentError";
  }
}

export class NoMatchupAvailableError extends TournamentError {
  constructor() {
    super("No unseen eligible matchup is available for this voter.", "NO_MATCHUP_AVAILABLE");
    this.name = "NoMatchupAvailableError";
  }
}

export class MatchupNotFoundError extends TournamentError {
  constructor() {
    super("The matchup does not exist.", "MATCHUP_NOT_FOUND");
    this.name = "MatchupNotFoundError";
  }
}

export class VoteRejectedError extends TournamentError {
  constructor(message: string, code = "VOTE_REJECTED") {
    super(message, code);
    this.name = "VoteRejectedError";
  }
}
