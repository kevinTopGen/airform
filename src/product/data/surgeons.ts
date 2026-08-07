import { localSurgeryCaseFixtures } from "../../tournament/data/fixtures";
import { adaptSurgeryCases } from "../../tournament/domain/externalContracts";
import type { SurgeonCase } from "../../tournament/domain/models";

export interface SurgeonProfile {
  readonly id: string;
  readonly slug: string;
  readonly name: string;
  readonly locationLabel: string;
  readonly latitude: number;
  readonly longitude: number;
  readonly procedures: readonly string[];
  readonly profileImageUrl?: string;
  /** Demo fallback until a persisted tournament score is available. */
  readonly communityScore: number;
  readonly bio: string;
  readonly specialties: readonly string[];
}

export interface PreviewNavigationDescriptor {
  readonly type: "navigate";
  readonly href: string;
  readonly surgeonId: string;
  readonly surgeonSlug: string;
}

export interface SurgeonProvider {
  listSurgeons(): readonly SurgeonProfile[];
  getSurgeonBySlug(slug: string): SurgeonProfile | null;
  getSurgeonCases(surgeonId: string): readonly SurgeonCase[];
  getSurgeonScore(surgeonId: string, liveScore?: number | null): number | null;
  launchPreview(surgeon: SurgeonProfile): PreviewNavigationDescriptor;
}

export const demoSurgeons: readonly SurgeonProfile[] = [
  {
    id: "surgeon-aurora",
    slug: "aurora",
    name: "Dr. Elena Aurora",
    locationLabel: "Coral Gables, Miami",
    latitude: 25.7215,
    longitude: -80.2684,
    procedures: ["Rhinoplasty", "Revision rhinoplasty"],
    communityScore: 94,
    bio: "A fictional Miami facial-plastics specialist focused on natural-looking profile refinement.",
    specialties: ["Natural profile refinement", "Preservation techniques"],
  },
  {
    id: "surgeon-boreal",
    slug: "boreal",
    name: "Dr. Mateo Boreal",
    locationLabel: "Brickell, Miami",
    latitude: 25.7617,
    longitude: -80.1918,
    procedures: ["Rhinoplasty", "Ethnic rhinoplasty"],
    communityScore: 91,
    bio: "A fictional surgeon known for individualized planning and balanced frontal-view results.",
    specialties: ["Frontal balance", "Individualized planning"],
  },
  {
    id: "surgeon-cascade",
    slug: "cascade",
    name: "Dr. Simone Cascade",
    locationLabel: "Miami Beach, Miami",
    latitude: 25.7907,
    longitude: -80.13,
    procedures: ["Rhinoplasty", "Functional rhinoplasty"],
    communityScore: 89,
    bio: "A fictional facial surgeon combining aesthetic goals with careful attention to breathing.",
    specialties: ["Functional outcomes", "Tip refinement"],
  },
  {
    id: "surgeon-meridian",
    slug: "meridian",
    name: "Dr. Adrian Meridian",
    locationLabel: "Coconut Grove, Miami",
    latitude: 25.7282,
    longitude: -80.2409,
    procedures: ["Rhinoplasty", "Revision rhinoplasty"],
    communityScore: 87,
    bio: "A fictional specialist offering conservative, structure-preserving rhinoplasty plans.",
    specialties: ["Conservative refinement", "Revision planning"],
  },
] as const;

// Keep the case-to-surgeon relationship authoritative in the tournament fixture seam.
const demoCases = adaptSurgeryCases(localSurgeryCaseFixtures);

export const listSurgeons = (): readonly SurgeonProfile[] => [...demoSurgeons];

export const getSurgeonBySlug = (slug: string): SurgeonProfile | null => {
  const normalizedSlug = slug.trim().toLowerCase();
  return demoSurgeons.find((surgeon) => surgeon.slug === normalizedSlug) ?? null;
};

export const getSurgeonCases = (surgeonId: string): readonly SurgeonCase[] =>
  demoCases.filter((surgeonCase) => surgeonCase.surgeonId === surgeonId);

export const getSurgeonScore = (
  surgeonId: string,
  liveScore?: number | null,
): number | null => {
  if (typeof liveScore === "number" && Number.isFinite(liveScore)) {
    return Math.min(100, Math.max(0, liveScore));
  }

  return demoSurgeons.find((surgeon) => surgeon.id === surgeonId)?.communityScore ?? null;
};

/** Pure navigation seam: callers decide whether and how to perform the navigation. */
export const launchPreview = (surgeon: SurgeonProfile): PreviewNavigationDescriptor => ({
  type: "navigate",
  href: `/surgeons/${encodeURIComponent(surgeon.slug)}/preview`,
  surgeonId: surgeon.id,
  surgeonSlug: surgeon.slug,
});

export const demoSurgeonProvider: SurgeonProvider = {
  listSurgeons,
  getSurgeonBySlug,
  getSurgeonCases,
  getSurgeonScore,
  launchPreview,
};
