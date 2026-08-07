import type { SurgeonCase } from "./models";

/** Exact shared record promised to the future host repository. */
export interface SurgeryCase {
  id: string;
  surgeonId: string;
  procedure: string;
  beforeImageUrl: string;
  afterImageUrl: string;
  view?: string;
  active?: boolean;
  metadata?: Record<string, unknown>;
}

const requireText = (value: string, field: string): string => {
  if (!value.trim()) {
    throw new Error(`SurgeryCase.${field} must be a non-empty string.`);
  }
  return value;
};

/**
 * The sole shape translation required when the authoritative host arrives.
 * Opaque IDs pass through unchanged; core code never parses or replaces them.
 */
export const adaptSurgeryCase = (source: SurgeryCase): SurgeonCase => {
  const id = requireText(source.id, "id");
  const surgeonId = requireText(source.surgeonId, "surgeonId");
  const procedureId = requireText(source.procedure, "procedure");
  const beforeImageUrl = requireText(source.beforeImageUrl, "beforeImageUrl");
  const afterImageUrl = requireText(source.afterImageUrl, "afterImageUrl");
  const view = source.view?.trim() || "profile";

  return {
    id,
    surgeonId,
    procedureId,
    view,
    status: source.active === false ? "inactive" : "active",
    images: {
      before: {
        id: `${id}:before`,
        url: beforeImageUrl,
        phase: "before",
        alt: `Anonymized ${view} view before procedure`,
      },
      after: {
        id: `${id}:after`,
        url: afterImageUrl,
        phase: "after",
        alt: `Anonymized ${view} view after procedure`,
      },
    },
    ...(source.metadata ? { metadata: source.metadata } : {}),
  };
};

export const adaptSurgeryCases = (sources: readonly SurgeryCase[]): readonly SurgeonCase[] =>
  sources.map(adaptSurgeryCase);
