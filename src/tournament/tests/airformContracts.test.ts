import { describe, expect, it } from "vitest";
import {
  adaptAirformSurgeonSignature,
  type AirformSurgeonSignatureJson,
} from "../integration/airformContracts";

const signature: AirformSurgeonSignatureJson = {
  id: "zhuravsky_ruslan",
  name: "Dr. Ruslan Zhuravsky, MD",
  tagline: "Example signature",
  n_pairs: 12,
  delta: {
    alar_width: -0.08,
    nasal_length: -0.01,
  },
  std: {
    alar_width: 0.02,
    nasal_length: 0.005,
  },
  delta_modes: {
    alar_width: "proportional",
    nasal_length: "absolute",
  },
};

describe("Airform SurgeonSignature adapter", () => {
  it("preserves the signature id exactly as the tournament surgeon id", () => {
    expect(adaptAirformSurgeonSignature(signature)).toEqual({
      id: signature.id,
      name: signature.name,
    });
  });

  it("rejects missing identity instead of inventing a fallback id or name", () => {
    expect(() =>
      adaptAirformSurgeonSignature({ ...signature, id: "  " }),
    ).toThrow("AirformSurgeonSignature.id");
    expect(() =>
      adaptAirformSurgeonSignature({ ...signature, name: "" }),
    ).toThrow("AirformSurgeonSignature.name");
  });
});
