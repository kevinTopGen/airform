import type { Surgeon } from "../domain/models";

export type AirformNoseField =
  | "alar_width"
  | "bridge_width"
  | "tip_width"
  | "nasal_length"
  | "dorsal_hump"
  | "tip_rotation_deg"
  | "tip_projection";

export type AirformDeltaMode = "absolute" | "proportional";

/** JSON emitted by Airform's NoseParams.to_dict(). None-valued fields are omitted. */
export type AirformNoseParamsJson = Readonly<
  Partial<Record<AirformNoseField, number>>
>;

/** Exact public JSON shape emitted by Airform's SurgeonSignature.to_dict(). */
export interface AirformSurgeonSignatureJson {
  readonly id: string;
  readonly name: string;
  readonly tagline: string;
  readonly n_pairs: number;
  readonly delta: AirformNoseParamsJson;
  readonly std: AirformNoseParamsJson;
  readonly delta_modes: Readonly<
    Partial<Record<AirformNoseField, AirformDeltaMode>>
  >;
}

const requireText = (value: string, field: "id" | "name"): string => {
  if (!value.trim()) {
    throw new Error(`AirformSurgeonSignature.${field} must be a non-empty string.`);
  }

  return value;
};

/**
 * Converts Airform's authoritative surgeon identity into the tournament model.
 *
 * The signature id passes through byte-for-byte because SurgeryCase.surgeonId
 * joins on this value. Simulation statistics remain owned by Airform and are
 * intentionally not copied into the tournament's surgeon/rating domain.
 */
export const adaptAirformSurgeonSignature = (
  source: AirformSurgeonSignatureJson,
): Surgeon => ({
  id: requireText(source.id, "id"),
  name: requireText(source.name, "name"),
});
