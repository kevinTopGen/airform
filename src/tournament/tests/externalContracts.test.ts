import { describe, expect, it } from "vitest";
import { adaptSurgeryCase, type SurgeryCase } from "../domain/externalContracts";

describe("SurgeryCase host adapter", () => {
  it("preserves opaque host IDs and maps the exact external contract", () => {
    const source: SurgeryCase = {
      id: "host_case::uuid/42",
      surgeonId: "host-surgeon-9",
      procedure: "rhinoplasty",
      beforeImageUrl: "https://host.example/signed/before",
      afterImageUrl: "https://host.example/signed/after",
      metadata: { hostOnly: { untouched: true } },
    };

    const adapted = adaptSurgeryCase(source);

    expect(adapted).toMatchObject({
      id: source.id,
      surgeonId: source.surgeonId,
      procedureId: source.procedure,
      view: "profile",
      status: "active",
      metadata: source.metadata,
    });
    expect(adapted.images.before.url).toBe(source.beforeImageUrl);
    expect(adapted.images.after.url).toBe(source.afterImageUrl);
  });

  it("marks active:false cases ineligible and rejects missing identity", () => {
    const inactive: SurgeryCase = {
      id: "case",
      surgeonId: "surgeon",
      procedure: "rhinoplasty",
      beforeImageUrl: "/before",
      afterImageUrl: "/after",
      active: false,
    };

    expect(adaptSurgeryCase(inactive).status).toBe("inactive");
    expect(() => adaptSurgeryCase({ ...inactive, id: "  " })).toThrow("SurgeryCase.id");
  });
});
