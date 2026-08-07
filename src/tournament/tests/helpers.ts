import type { CaseRating, SurgeonCase } from "../domain/models";

export const makeCase = (
  id: string,
  surgeonId: string,
  view = "profile",
): SurgeonCase => ({
  id,
  surgeonId,
  procedureId: "rhinoplasty",
  view,
  status: "active",
  images: {
    before: { id: `${id}-b`, url: `/before/${id}`, phase: "before", alt: "Before" },
    after: { id: `${id}-a`, url: `/after/${id}`, phase: "after", alt: "After" },
  },
});

export const makeRating = (
  caseId: string,
  rating: number,
  comparisons: number,
  wins = Math.ceil(comparisons / 2),
): CaseRating => ({
  caseId,
  rating,
  comparisons,
  wins,
  losses: comparisons - wins,
  updatedAt: comparisons ? "2026-08-06T12:00:00.000Z" : null,
});
