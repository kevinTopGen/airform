import type { CaseId, Surgeon, SurgeonCase } from "../domain/models";
import type { CaseQuery, SurgeonCaseProvider } from "../domain/ports";

export class LocalFixtureCaseProvider implements SurgeonCaseProvider {
  readonly #cases: readonly SurgeonCase[];
  readonly #surgeons: readonly Surgeon[];

  constructor(cases: readonly SurgeonCase[], surgeons?: readonly Surgeon[]) {
    this.#cases = [...cases];
    this.#surgeons = surgeons
      ? [...surgeons]
      : [...new Set(cases.map((surgeonCase) => surgeonCase.surgeonId))].map((id) => ({
          id,
          name: id,
        }));
  }

  async getSurgeon(surgeonId: string): Promise<Surgeon | null> {
    return this.#surgeons.find((surgeon) => surgeon.id === surgeonId) ?? null;
  }

  async listSurgeons(): Promise<readonly Surgeon[]> {
    return [...this.#surgeons];
  }

  async listEligibleCases(query: CaseQuery = {}): Promise<readonly SurgeonCase[]> {
    const { procedureId, view, activeOnly = true } = query;

    return this.#cases.filter(
      (surgeonCase) =>
        (!activeOnly || surgeonCase.status === "active") &&
        (!procedureId || surgeonCase.procedureId === procedureId) &&
        (!view || surgeonCase.view === view),
    );
  }

  async getCase(caseId: CaseId): Promise<SurgeonCase | null> {
    return this.#cases.find((surgeonCase) => surgeonCase.id === caseId) ?? null;
  }
}
