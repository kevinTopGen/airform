import type { Surgeon, SurgeonScore } from "../domain/models";
import { ProfileIcon } from "./icons";
import { SurgeonCommunityScore } from "./SurgeonCommunityScore";

interface LeaderboardProps {
  scores: readonly SurgeonScore[];
  surgeons: readonly Surgeon[];
  expanded?: boolean;
}

export function Leaderboard({ scores, surgeons, expanded = false }: LeaderboardProps) {
  const names = new Map(surgeons.map((surgeon) => [surgeon.id, surgeon.name]));
  const visible = expanded ? scores : scores.slice(0, 5);

  return (
    <ol className={`leaderboard-list${expanded ? " leaderboard-list--expanded" : ""}`}>
      {visible.map((score, index) => (
        <li key={score.surgeonId}>
          <span className="leaderboard-list__rank">{index + 1}</span>
          <ProfileIcon className="leaderboard-list__avatar" />
          <span className="leaderboard-list__name">{names.get(score.surgeonId) ?? "Community surgeon"}</span>
          <SurgeonCommunityScore score={score} compact />
        </li>
      ))}
      {visible.length === 0 && <li className="leaderboard-list__empty">Cast the first vote to begin the standings.</li>}
    </ol>
  );
}
