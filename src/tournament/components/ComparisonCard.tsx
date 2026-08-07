import type { AnonymousCase } from "../domain/models";
import { ArrowIcon } from "./icons";

interface ComparisonCardProps {
  caseData: AnonymousCase;
  label: "A" | "B";
  disabled?: boolean;
  selected?: boolean;
  onSelect: (caseId: string) => void;
}

export function ComparisonCard({
  caseData,
  label,
  disabled = false,
  selected = false,
  onSelect,
}: ComparisonCardProps) {
  return (
    <button
      type="button"
      className={`comparison-card${selected ? " is-selected" : ""}`}
      onClick={() => onSelect(caseData.id)}
      disabled={disabled}
      aria-label={`Choose result ${label}`}
      aria-keyshortcuts={label.toLowerCase()}
    >
      <span className="comparison-card__head">
        <span className="comparison-card__letter" aria-hidden="true">{label}</span>
        <span className="comparison-card__action">
          {selected ? "Preference recorded" : "Select to choose"}
          <ArrowIcon />
        </span>
      </span>
      <span className="comparison-card__rule" />
      <span className="comparison-card__labels" aria-hidden="true">
        <span>Before</span>
        <span>After</span>
      </span>
      <span className="comparison-card__images">
        <img src={caseData.images.before.url} alt={caseData.images.before.alt} />
        <img src={caseData.images.after.url} alt={caseData.images.after.alt} />
      </span>
    </button>
  );
}
