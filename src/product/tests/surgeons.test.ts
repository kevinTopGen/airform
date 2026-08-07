import { describe, expect, it } from "vitest";
import { localSurgeryCaseFixtures } from "../../tournament/data/fixtures";
import {
  demoSurgeons,
  getSurgeonBySlug,
  getSurgeonCases,
  getSurgeonScore,
  launchPreview,
  listSurgeons,
} from "../data/surgeons";

describe("demo surgeon provider", () => {
  it("lists exactly four stable, mappable Miami-area profiles", () => {
    expect(listSurgeons().map((surgeon) => surgeon.id)).toEqual([
      "surgeon-aurora",
      "surgeon-boreal",
      "surgeon-cascade",
      "surgeon-meridian",
    ]);
    expect(demoSurgeons.every((surgeon) => Number.isFinite(surgeon.latitude))).toBe(true);
    expect(demoSurgeons.every((surgeon) => Number.isFinite(surgeon.longitude))).toBe(true);
  });

  it("looks profiles up by normalized slug", () => {
    expect(getSurgeonBySlug(" AURORA ")?.id).toBe("surgeon-aurora");
    expect(getSurgeonBySlug("missing")).toBeNull();
  });

  it("derives case ownership from the canonical surgery fixtures", () => {
    const expectedIds = localSurgeryCaseFixtures
      .filter((surgeonCase) => surgeonCase.surgeonId === "surgeon-aurora")
      .map((surgeonCase) => surgeonCase.id);

    expect(getSurgeonCases("surgeon-aurora").map((surgeonCase) => surgeonCase.id)).toEqual(
      expectedIds,
    );
    expect(getSurgeonCases("surgeon-meridian")).toEqual([]);
  });

  it("prefers a live score and otherwise uses the profile fallback", () => {
    expect(getSurgeonScore("surgeon-aurora", 82.5)).toBe(82.5);
    expect(getSurgeonScore("surgeon-aurora")).toBe(94);
    expect(getSurgeonScore("unknown")).toBeNull();
  });

  it("returns a navigation descriptor without performing navigation", () => {
    const surgeon = getSurgeonBySlug("cascade");
    expect(surgeon && launchPreview(surgeon)).toEqual({
      type: "navigate",
      href: "/surgeons/cascade/preview",
      surgeonId: "surgeon-cascade",
      surgeonSlug: "cascade",
    });
  });
});
