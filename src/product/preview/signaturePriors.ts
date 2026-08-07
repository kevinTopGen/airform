export type AirformArchetypeId = "conservative" | "alar" | "dorsal" | "tip" | "signature";

export interface SignaturePrior {
  readonly id: AirformArchetypeId;
  readonly name: string;
  readonly tagline: string;
  readonly delta: {
    readonly alarWidth: number;
    readonly bridgeWidth: number;
    readonly tipWidth: number;
    readonly nasalLength: number;
  };
}

/** Browser-safe mirrors of nosesim.signatures.ARCHETYPES for the static demo. */
export const signaturePriors: Readonly<Record<AirformArchetypeId, SignaturePrior>> = {
  conservative: {
    id: "conservative",
    name: "The Conservative",
    tagline: "Preserves character. You, lightly edited.",
    delta: { alarWidth: -0.023, bridgeWidth: -0.041, tipWidth: -0.049, nasalLength: -0.005 },
  },
  alar: {
    id: "alar",
    name: "The Base Reducer",
    tagline: "Narrows the base while leaving the bridge restrained.",
    delta: { alarWidth: -0.095, bridgeWidth: -0.014, tipWidth: -0.055, nasalLength: 0 },
  },
  dorsal: {
    id: "dorsal",
    name: "The Dorsal Preservationist",
    tagline: "Refines the bridge while preserving the tip.",
    delta: { alarWidth: -0.015, bridgeWidth: -0.157, tipWidth: -0.033, nasalLength: -0.008 },
  },
  tip: {
    id: "tip",
    name: "The Tip Sculptor",
    tagline: "Concentrates refinement below the supratip.",
    delta: { alarWidth: -0.034, bridgeWidth: -0.027, tipWidth: -0.151, nasalLength: -0.045 },
  },
  signature: {
    id: "signature",
    name: "The Miami Signature",
    tagline: "A more visible, full-nose refinement.",
    delta: { alarWidth: -0.107, bridgeWidth: -0.163, tipWidth: -0.192, nasalLength: -0.038 },
  },
};

export const getSignaturePrior = (id: AirformArchetypeId): SignaturePrior => signaturePriors[id];

