import type { SurgeryCase } from "../domain/externalContracts";
import { adaptSurgeryCases } from "../domain/externalContracts";
import type { Surgeon } from "../domain/models";

export const localFixtureSurgeons: readonly Surgeon[] = [
  { id: "surgeon-aurora", name: "Dr. Aurora", slug: "aurora" },
  { id: "surgeon-boreal", name: "Dr. Boreal", slug: "boreal" },
  { id: "surgeon-cascade", name: "Dr. Cascade", slug: "cascade" },
];

/** Uses synthetic local demo art; no scraped dataset enters the application. */
export const localSurgeryCaseFixtures: readonly SurgeryCase[] = [
  {
    id: "case-01",
    surgeonId: "surgeon-aurora",
    procedure: "rhinoplasty",
    beforeImageUrl: "/fixtures/case-01-before.webp",
    afterImageUrl: "/fixtures/case-01-after.webp",
    view: "profile",
    active: true,
  },
  {
    id: "case-02",
    surgeonId: "surgeon-aurora",
    procedure: "rhinoplasty",
    beforeImageUrl: "/fixtures/case-02-before.webp",
    afterImageUrl: "/fixtures/case-02-after.webp",
    view: "profile",
    active: true,
  },
  {
    id: "case-03",
    surgeonId: "surgeon-boreal",
    procedure: "rhinoplasty",
    beforeImageUrl: "/fixtures/case-03-before.webp",
    afterImageUrl: "/fixtures/case-03-after.webp",
    view: "profile",
    active: true,
  },
  {
    id: "case-04",
    surgeonId: "surgeon-cascade",
    procedure: "rhinoplasty",
    beforeImageUrl: "/fixtures/case-04-before.webp",
    afterImageUrl: "/fixtures/case-04-after.webp",
    view: "profile",
    active: true,
  },
];

export const localFixtureCases = adaptSurgeryCases(localSurgeryCaseFixtures);
