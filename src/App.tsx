import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnonymousMatchup, Surgeon, SurgeonScore } from "./tournament/domain/models";
import { ComparisonCard } from "./tournament/components/ComparisonCard";
import { ArrowIcon, PeopleIcon, ProfileIcon, TrendIcon } from "./tournament/components/icons";
import { Leaderboard } from "./tournament/components/Leaderboard";
import {
  beginNewAnonymousSession,
  browserCaseProvider,
  browserRepository,
  createIdempotencyKey,
  getOrCreateVoterId,
  resumeOpenMatchup,
  tournamentService,
} from "./tournament/ui/browserTournament";

type View = "compare" | "leaderboard";

interface AppProps {
  readonly onExit?: () => void;
}

export function App({ onExit }: AppProps = {}) {
  const [view, setView] = useState<View>("compare");
  const [voterId, setVoterId] = useState(getOrCreateVoterId);
  const [matchup, setMatchup] = useState<AnonymousMatchup | null>(null);
  const [scores, setScores] = useState<readonly SurgeonScore[]>([]);
  const [surgeons, setSurgeons] = useState<readonly Surgeon[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "voting" | "complete" | "error">("loading");
  const [message, setMessage] = useState("");
  const [sessionVotes, setSessionVotes] = useState(() => browserRepository.getVoterVoteCount(voterId));
  const initializedVoterId = useRef<string | null>(null);

  const refreshScores = useCallback(async () => {
    const [nextScores, nextSurgeons] = await Promise.all([
      tournamentService.getSurgeonScores("rhinoplasty"),
      browserCaseProvider.listSurgeons(),
    ]);
    setScores(nextScores);
    setSurgeons(nextSurgeons);
  }, []);

  const loadNextMatchup = useCallback(async (activeVoterId: string, resume = false) => {
    setStatus("loading");
    setMessage("");
    setSelectedCaseId(null);
    try {
      if (resume) {
        const existing = await resumeOpenMatchup(activeVoterId);
        if (existing) {
          setMatchup(existing);
          setStatus("ready");
          return;
        }
      }
      const next = await tournamentService.getNextMatchup({
        voterId: activeVoterId,
        procedureId: "rhinoplasty",
        view: "profile",
      });
      setMatchup(next);
      setStatus("ready");
    } catch {
      setMatchup(null);
      setStatus("complete");
      setMessage("You’ve seen every available pairing in this round.");
    }
  }, []);

  useEffect(() => {
    // React StrictMode intentionally replays mount effects in development.
    // Avoid creating and persisting two matchups for the same initial screen.
    if (initializedVoterId.current === voterId) return;
    initializedVoterId.current = voterId;
    void Promise.all([loadNextMatchup(voterId, true), refreshScores()]);
  }, [loadNextMatchup, refreshScores, voterId]);

  const submitVote = useCallback(async (caseId: string) => {
    if (!matchup || status !== "ready") return;
    setSelectedCaseId(caseId);
    setStatus("voting");
    setMessage("Preference recorded. Preparing the next pair…");
    try {
      await tournamentService.submitVote({
        matchupId: matchup.id,
        voterId,
        selectedCaseId: caseId,
        idempotencyKey: createIdempotencyKey(),
      });
      setSessionVotes(browserRepository.getVoterVoteCount(voterId));
      await refreshScores();
      window.setTimeout(() => void loadNextMatchup(voterId), 280);
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "We could not record that preference.");
    }
  }, [loadNextMatchup, matchup, refreshScores, status, voterId]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (view !== "compare" || status !== "ready" || !matchup) return;
      if (event.key.toLowerCase() === "a") void submitVote(matchup.caseA.id);
      if (event.key.toLowerCase() === "b") void submitVote(matchup.caseB.id);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [matchup, status, submitVote, view]);

  const startNewRound = () => {
    const nextVoterId = beginNewAnonymousSession();
    setVoterId(nextVoterId);
    setSessionVotes(0);
  };

  const skipMatchup = () => {
    if (!matchup || status !== "ready") return;
    browserRepository.expireMatchup(matchup.id);
    void loadNextMatchup(voterId);
  };

  const totalComparisons = browserRepository.getVoteCount();
  const rankedScores = useMemo(
    () => [...scores].sort((a, b) => b.rating - a.rating),
    [scores],
  );

  return (
    <div className="site-shell" data-airform-surface="tournament">
      <header className="site-header">
        <button className="wordmark" type="button" onClick={() => onExit ? onExit() : setView("compare")}>
          airform <span>/</span> compare
        </button>
        <nav aria-label="Primary navigation">
          {onExit ? <button onClick={onExit}>Find surgeons</button> : null}
          <button className={view === "compare" ? "is-active" : ""} onClick={() => setView("compare")}>Compare</button>
          <button className={view === "leaderboard" ? "is-active" : ""} onClick={() => setView("leaderboard")}>Leaderboard</button>
        </nav>
        <div className="anonymous-mark" aria-label="Anonymous participant"><ProfileIcon /></div>
      </header>

      {view === "compare" ? (
        <main className="compare-layout">
          <section className="tournament-stage" aria-labelledby="comparison-heading">
            <div className="community-count"><PeopleIcon /> {totalComparisons.toLocaleString()} community comparisons</div>
            <h1 id="comparison-heading">Which result feels more balanced?</h1>
            <p className="lede">Choose A or B by selecting the entire side you prefer.</p>

            <div className="comparison-region" aria-live="polite" aria-busy={status === "loading" || status === "voting"}>
              {matchup ? (
                <>
                  <ComparisonCard caseData={matchup.caseA} label="A" disabled={status !== "ready"} selected={selectedCaseId === matchup.caseA.id} onSelect={submitVote} />
                  <span className="versus" aria-hidden="true">VS</span>
                  <ComparisonCard caseData={matchup.caseB} label="B" disabled={status !== "ready"} selected={selectedCaseId === matchup.caseB.id} onSelect={submitVote} />
                </>
              ) : (
                <div className="empty-state">
                  <span>{status === "loading" ? "Selecting a balanced pair…" : message}</span>
                  {status === "complete" && <button type="button" onClick={startNewRound}>Begin a fresh round <ArrowIcon /></button>}
                </div>
              )}
            </div>
            {matchup && <p className={`status-line${status === "error" ? " is-error" : ""}`}>{message}</p>}
            {matchup && status === "ready" && (
              <div className="skip-row">
                <span>Can’t decide?</span>
                <button type="button" onClick={skipMatchup}>Next pair <ArrowIcon /></button>
              </div>
            )}
          </section>

          <aside className="standings-panel" aria-labelledby="standings-heading">
            <h2 id="standings-heading">Community standings</h2>
            <Leaderboard scores={rankedScores} surgeons={surgeons} />
            <button className="text-link" type="button" onClick={() => setView("leaderboard")}>View full leaderboard <ArrowIcon /></button>
            <div className="standings-stats">
              <p><PeopleIcon /><span>Session votes</span><strong>{sessionVotes}</strong></p>
              <p><TrendIcon /><span>All comparisons</span><strong>{totalComparisons.toLocaleString()}</strong></p>
              <p><ProfileIcon /><span>Participation</span><strong>Anonymous</strong></p>
            </div>
          </aside>
        </main>
      ) : (
        <main className="leaderboard-page">
          <div className="community-count"><PeopleIcon /> Community-ranked outcomes</div>
          <div className="leaderboard-page__heading">
            <div><h1>Community standings</h1><p>Preference scores based only on anonymous before-and-after comparisons.</p></div>
            <button type="button" onClick={() => setView("compare")}>Return to compare <ArrowIcon /></button>
          </div>
          <Leaderboard scores={rankedScores} surgeons={surgeons} expanded />
        </main>
      )}

      <footer>Aesthetic preference only. Not medical advice or a measure of surgical safety.</footer>
    </div>
  );
}
