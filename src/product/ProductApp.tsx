import { FormEvent, useEffect, useMemo, useState } from "react";
import { App as TournamentExperience } from "../App";
import type { SurgeonCase } from "../tournament/domain/models";
import { tournamentService } from "../tournament/ui/browserTournament";
import { SurgeonMap } from "./map/SurgeonMap";
import { demoSurgeons, getSurgeonBySlug, getSurgeonCases, type SurgeonProfile } from "./data/surgeons";
import { BrowserPreview } from "./preview/BrowserPreview";
import "./product-shell.css";

type Route = { name: "home" } | { name: "profile"; slug: string } | { name: "preview"; slug: string } | { name: "tournament" };

const APP_BASE = import.meta.env.BASE_URL.replace(/\/$/, "");
const publicAsset = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\//, "")}`;

const parseRoute = (): Route => {
  const unbasedPath = APP_BASE && window.location.pathname.startsWith(APP_BASE)
    ? window.location.pathname.slice(APP_BASE.length)
    : window.location.pathname;
  const path = unbasedPath.replace(/\/+$/, "") || "/";
  if (path === "/tournament") return { name: "tournament" };
  const profile = path.match(/^\/surgeons\/([^/]+)$/);
  if (profile?.[1]) return { name: "profile", slug: decodeURIComponent(profile[1]) };
  const preview = path.match(/^\/preview\/([^/]+)$/);
  if (preview?.[1]) return { name: "preview", slug: decodeURIComponent(preview[1]) };
  return { name: "home" };
};

const ArrowIcon = () => <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 10h13M11 5l5 5-5 5" /></svg>;
const SearchIcon = () => <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" /><path d="m15.5 15.5 5 5" /></svg>;
const PinIcon = () => <svg viewBox="0 0 18 18" aria-hidden="true"><path d="M9 16s5-4.6 5-9A5 5 0 0 0 4 7c0 4.4 5 9 5 9Z" /><circle cx="9" cy="7" r="1.7" /></svg>;

interface NavigateProps { navigate: (path: string) => void }

function ProductHeader({ navigate }: NavigateProps) {
  return <header className="product-header">
    <button className="product-wordmark" type="button" onClick={() => navigate("/")}>airform <span>/</span></button>
    <nav aria-label="Primary navigation">
      <button type="button" onClick={() => navigate("/#surgeons")}>Find surgeons</button>
      <button type="button" onClick={() => navigate("/tournament")}>Community ranking</button>
      <button type="button" onClick={() => navigate("/#how-it-works")}>How it works</button>
    </nav>
    <button className="header-cta" type="button" onClick={() => navigate(`/preview/${demoSurgeons[0].slug}`)}>Preview on me</button>
  </header>;
}

function CasePair({ caseData }: { caseData: SurgeonCase }) {
  return <div className="case-pair">
    <figure><img src={publicAsset(caseData.images.before.url)} alt={caseData.images.before.alt} /><figcaption>Before</figcaption></figure>
    <figure><img src={publicAsset(caseData.images.after.url)} alt={caseData.images.after.alt} /><figcaption>After</figcaption></figure>
  </div>;
}

function SurgeonRow({ surgeon, rank, selected, onSelect, onOpen }: { surgeon: SurgeonProfile; rank: number; selected: boolean; onSelect: () => void; onOpen: () => void }) {
  const firstCase = getSurgeonCases(surgeon.id)[0];
  return <article className={`surgeon-row${selected ? " is-selected" : ""}`} onMouseEnter={onSelect}>
    <button className="surgeon-row__main" type="button" onClick={onSelect} aria-pressed={selected}>
      <span className="surgeon-rank">{rank}</span>
      <span className="surgeon-avatar" aria-hidden="true">{surgeon.name.split(" ").at(-1)?.[0]}</span>
      <span className="surgeon-identity"><strong>{surgeon.name}</strong><small><PinIcon /> {surgeon.locationLabel}</small><small>{surgeon.procedures[0]}</small></span>
      {firstCase ? <span className="surgeon-result" aria-hidden="true"><img src={publicAsset(firstCase.images.before.url)} alt="" /><img src={publicAsset(firstCase.images.after.url)} alt="" /></span> : <span className="surgeon-result is-empty" aria-hidden="true" />}
      <span className="surgeon-score"><strong>{surgeon.communityScore}</strong><small>/100</small></span>
    </button>
    <button className="surgeon-row__open" type="button" onClick={onOpen} aria-label={`View ${surgeon.name}'s profile`}><ArrowIcon /></button>
  </article>;
}

function HomePage({ navigate, surgeons }: NavigateProps & { surgeons: readonly SurgeonProfile[] }) {
  const [selectedId, setSelectedId] = useState(surgeons[0].id);
  const selectedSurgeon = useMemo(() => surgeons.find((surgeon) => surgeon.id === selectedId) ?? surgeons[0], [selectedId, surgeons]);
  const scrollToSurgeons = (event?: FormEvent) => { event?.preventDefault(); document.getElementById("surgeons")?.scrollIntoView({ behavior: "smooth" }); };
  return <>
    <section className="product-hero">
      <div className="hero-copy"><h1>Find the surgeon<br />whose work feels like you.</h1><p>Explore real outcomes, compare community preferences, and preview a surgeon’s aesthetic on your own face.</p>
        <form className="location-search" onSubmit={scrollToSurgeons}><label><SearchIcon /><span className="sr-only">Location</span><input defaultValue="Miami, FL" aria-label="Miami, FL or ZIP code" /></label><button type="submit">Explore Miami surgeons</button></form>
        <button className="hero-text-link" type="button" onClick={() => navigate("/tournament")}>Rank results <ArrowIcon /></button>
      </div>
      <div className="hero-visual" aria-label="Fictional profile outcome preview"><img className="hero-profile hero-profile--ghost-two" src={publicAsset("fixtures/case-01-before.webp")} alt="" /><img className="hero-profile hero-profile--ghost-one" src={publicAsset("fixtures/case-01-before.webp")} alt="" /><img className="hero-profile hero-profile--main" src={publicAsset("fixtures/case-01-after.webp")} alt="Fictional anonymized rhinoplasty profile result" /><span className="hero-measure hero-measure--horizontal" /><span className="hero-measure hero-measure--vertical" /></div>
    </section>
    <section className="discovery-section" id="surgeons">
      <div className="discovery-heading"><div><h2>Four perspectives. One city.</h2><p>Community scores reflect anonymous preference rankings.</p></div><p>Showing four rhinoplasty surgeons in Miami, FL</p></div>
      <div className="discovery-grid"><div className="surgeon-rail">{surgeons.map((surgeon, index) => <SurgeonRow key={surgeon.id} surgeon={surgeon} rank={index + 1} selected={surgeon.id === selectedSurgeon.id} onSelect={() => setSelectedId(surgeon.id)} onOpen={() => navigate(`/surgeons/${surgeon.slug}`)} />)}<button className="rail-link" type="button" onClick={() => navigate("/tournament")}>View full rankings <ArrowIcon /></button></div>
        <SurgeonMap surgeons={surgeons} selectedSurgeonId={selectedSurgeon.id} onSelectSurgeon={setSelectedId} onViewProfile={(slug) => navigate(`/surgeons/${slug}`)} />
      </div>
    </section>
    <section className="ranking-callout" id="how-it-works"><div className="ranking-symbol" aria-hidden="true">A/B</div><div><h2>Your preference shapes the score.</h2><p>Compare outcomes side by side and help others understand the community’s aesthetic preference.</p></div><button type="button" onClick={() => navigate("/tournament")}>Start comparing</button></section>
  </>;
}

function ProfilePage({ slug, navigate, scores }: NavigateProps & { slug: string; scores: ReadonlyMap<string, number> }) {
  const storedSurgeon = getSurgeonBySlug(slug);
  const surgeon = storedSurgeon ? { ...storedSurgeon, communityScore: scores.get(storedSurgeon.id) ?? storedSurgeon.communityScore } : null;
  if (!surgeon) return <NotFound navigate={navigate} />;
  const cases = getSurgeonCases(surgeon.id);
  return <main className="profile-page">
    <button className="back-link" type="button" onClick={() => navigate("/#surgeons")}><ArrowIcon /> All Miami surgeons</button>
    <section className="profile-hero"><div className="profile-monogram" aria-hidden="true">{surgeon.name.split(" ").at(-1)?.[0]}</div><div className="profile-title"><h1>{surgeon.name}</h1><p><PinIcon /> {surgeon.locationLabel}</p><p>{surgeon.specialties.join(" · ")}</p></div><div className="profile-score"><span>Community score</span><strong>{surgeon.communityScore}<small>/100</small></strong><p>From anonymous outcome comparisons</p></div><button className="profile-preview" type="button" onClick={() => navigate(`/preview/${surgeon.slug}`)}>Preview yourself with this surgeon <ArrowIcon /></button></section>
    <section className="profile-results"><div className="section-title"><h2>Before / after results</h2><p>Fictional demonstration cases for interface testing.</p></div>{cases.length ? <div className="profile-cases">{cases.map((caseData) => <CasePair key={caseData.id} caseData={caseData} />)}</div> : <p className="profile-empty">Case imagery will connect through the shared case provider.</p>}</section>
    <section className="profile-details"><div><h2>About</h2><p>{surgeon.bio}</p></div><div><h2>Location</h2><p>{surgeon.locationLabel}</p><SurgeonMap surgeons={[surgeon]} selectedSurgeonId={surgeon.id} onSelectSurgeon={() => undefined} onViewProfile={() => undefined} /></div></section>
  </main>;
}

function PreviewExperience({ slug, navigate }: NavigateProps & { slug: string }) {
  const surgeon = getSurgeonBySlug(slug);
  if (!surgeon) return <NotFound navigate={navigate} />;
  return <BrowserPreview surgeon={surgeon} navigate={navigate} />;
}

function NotFound({ navigate }: NavigateProps) { return <main className="not-found"><h1>That surgeon isn’t in this demo.</h1><button type="button" onClick={() => navigate("/")}>Return home</button></main>; }

export function ProductApp() {
  const [route, setRoute] = useState<Route>(parseRoute);
  const [communityScores, setCommunityScores] = useState<ReadonlyMap<string, number>>(new Map());
  useEffect(() => { const onPopState = () => setRoute(parseRoute()); window.addEventListener("popstate", onPopState); return () => window.removeEventListener("popstate", onPopState); }, []);
  useEffect(() => {
    void tournamentService.getSurgeonScores("rhinoplasty").then((scores) => {
      setCommunityScores(new Map(scores.flatMap((score) => score.score == null ? [] : [[score.surgeonId, Math.round(score.score)] as const])));
    });
  }, [route.name]);
  const navigate = (path: string) => { const [pathname, hash] = path.split("#"); const nextPath = pathname || window.location.pathname; window.history.pushState({}, "", `${APP_BASE}${nextPath}${hash ? `#${hash}` : ""}`); setRoute(parseRoute()); window.scrollTo({ top: 0, behavior: "smooth" }); if (hash) window.setTimeout(() => document.getElementById(hash)?.scrollIntoView({ behavior: "smooth" }), 0); };
  if (route.name === "tournament") return <TournamentExperience onExit={() => navigate("/")} />;
  const scoredSurgeons = demoSurgeons.map((surgeon) => ({ ...surgeon, communityScore: communityScores.get(surgeon.id) ?? surgeon.communityScore }));
  return <div className="product-shell"><ProductHeader navigate={navigate} />{route.name === "home" ? <HomePage navigate={navigate} surgeons={scoredSurgeons} /> : null}{route.name === "profile" ? <ProfilePage slug={route.slug} navigate={navigate} scores={communityScores} /> : null}{route.name === "preview" ? <PreviewExperience slug={route.slug} navigate={navigate} /> : null}<footer className="product-footer"><span>airform / Miami demo</span><span>Aesthetic exploration only. Not medical advice.</span></footer></div>;
}

export default ProductApp;
