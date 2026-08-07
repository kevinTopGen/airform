import type { SurgeonScore } from "../domain/models";

export interface SurgeonCommunityScoreProps {
  score: SurgeonScore | null;
  compact?: boolean;
  className?: string;
}

/** Reusable profile-page score: intentionally labels preference, never quality or safety. */
export function SurgeonCommunityScore({
  score,
  compact = false,
  className = "",
}: SurgeonCommunityScoreProps) {
  const ready = score && !score.provisional && score.score !== null;
  const displayScore = ready ? Math.round(score.score!) : null;

  return (
    <section className={`surgeon-score ${compact ? "surgeon-score--compact" : ""} ${className}`.trim()}>
      <span className="surgeon-score__eyebrow">Community preference</span>
      <div className="surgeon-score__value">
        {displayScore === null ? <span className="surgeon-score__pending">Gathering votes</span> : displayScore}
        {displayScore !== null && <span aria-hidden="true">/100</span>}
      </div>
      {!compact && (
        <p>
          {score
            ? `${score.comparisons.toLocaleString()} aesthetic comparison${score.comparisons === 1 ? "" : "s"}`
            : "Not enough votes yet."}
        </p>
      )}
    </section>
  );
}
