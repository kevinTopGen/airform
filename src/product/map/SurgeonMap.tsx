/// <reference types="google.maps" />

import { useEffect, useMemo, useRef, useState } from "react";
import "./surgeon-map.css";

export interface SurgeonMapSurgeon {
  id: string;
  slug: string;
  name: string;
  latitude: number;
  longitude: number;
  locationLabel: string;
  communityScore: number;
  profileImageUrl?: string;
}

export interface SurgeonMapProps {
  surgeons: readonly SurgeonMapSurgeon[];
  selectedSurgeonId: string | null;
  onSelectSurgeon: (id: string) => void;
  onViewProfile: (slug: string) => void;
}

type MapLoadState = "loading" | "ready" | "error";

const SCRIPT_ID = "airform-google-maps-script";
const MIAMI_CENTER = { lat: 25.778, lng: -80.202 };
const MIAMI_BOUNDS = {
  north: 25.98,
  east: -80.08,
  south: 25.55,
  west: -80.36,
};

let googleMapsPromise: Promise<typeof google.maps> | null = null;

function loadGoogleMaps(apiKey: string): Promise<typeof google.maps> {
  if (window.google?.maps) return Promise.resolve(window.google.maps);
  if (googleMapsPromise) return googleMapsPromise;

  const pending = new Promise<typeof google.maps>((resolve, reject) => {
    const finish = () => {
      if (window.google?.maps) resolve(window.google.maps);
      else reject(new Error("Google Maps loaded without the Maps API."));
    };
    const fail = () => reject(new Error("Google Maps could not be loaded."));
    const existing = document.getElementById(SCRIPT_ID) as HTMLScriptElement | null;

    if (existing) {
      existing.addEventListener("load", finish, { once: true });
      existing.addEventListener("error", fail, { once: true });
      return;
    }

    const script = document.createElement("script");
    script.id = SCRIPT_ID;
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&v=weekly&loading=async`;
    script.async = true;
    script.defer = true;
    script.addEventListener("load", finish, { once: true });
    script.addEventListener("error", fail, { once: true });
    document.head.appendChild(script);
  });
  const retryable = pending.catch((error: unknown) => {
    googleMapsPromise = null;
    throw error;
  });
  googleMapsPromise = retryable;

  return retryable;
}

function markerIcon(isSelected: boolean): google.maps.Icon {
  const width = isSelected ? 58 : 50;
  const height = isSelected ? 68 : 59;
  const border = isSelected ? "%23f5dfc4" : "%23ffffff";
  const glow = isSelected ? "filter='drop-shadow(0 7px 7px rgba(57,12,22,.35))'" : "";
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${width}' height='${height}' viewBox='0 0 58 68'><g ${glow}><path d='M29 2C14.1 2 4 12.1 4 26.4 4 45.1 29 66 29 66s25-20.9 25-39.6C54 12.1 43.9 2 29 2Z' fill='%237c1731' stroke='${border}' stroke-width='${isSelected ? 4 : 2.5}'/><circle cx='29' cy='26' r='16.5' fill='%239b2845'/><path d='M18 17.5c6-5.8 15.5-6.4 22.1-1.2' fill='none' stroke='%23ffffff' stroke-opacity='.2' stroke-width='2' stroke-linecap='round'/></g></svg>`;

  return {
    url: `data:image/svg+xml;charset=UTF-8,${svg}`,
    scaledSize: new google.maps.Size(width, height),
    anchor: new google.maps.Point(width / 2, height),
    labelOrigin: new google.maps.Point(width / 2, isSelected ? 28 : 24),
  };
}

function scoreLabel(score: number): google.maps.MarkerLabel {
  return {
    text: String(Math.round(score)),
    color: "#ffffff",
    fontFamily: '"Helvetica Neue", sans-serif',
    fontSize: "12px",
    fontWeight: "800",
  };
}

function initials(name: string) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("");
}

function deterministicPosition(surgeon: SurgeonMapSurgeon) {
  const isInMiami =
    surgeon.latitude >= MIAMI_BOUNDS.south &&
    surgeon.latitude <= MIAMI_BOUNDS.north &&
    surgeon.longitude >= MIAMI_BOUNDS.west &&
    surgeon.longitude <= MIAMI_BOUNDS.east;

  if (isInMiami) {
    return {
      left: ((surgeon.longitude - MIAMI_BOUNDS.west) / (MIAMI_BOUNDS.east - MIAMI_BOUNDS.west)) * 100,
      top: (1 - (surgeon.latitude - MIAMI_BOUNDS.south) / (MIAMI_BOUNDS.north - MIAMI_BOUNDS.south)) * 100,
    };
  }

  const hash = [...surgeon.id].reduce((value, character) => (value * 31 + character.charCodeAt(0)) >>> 0, 2166136261);
  return { left: 18 + (hash % 64), top: 18 + ((hash >>> 7) % 58) };
}

function SurgeonPopover({ surgeon, onViewProfile }: { surgeon: SurgeonMapSurgeon; onViewProfile: (slug: string) => void }) {
  return (
    <aside className="surgeon-map__popover" aria-live="polite" aria-label={`${surgeon.name} map details`}>
      <div className="surgeon-map__portrait" aria-hidden="true">
        {surgeon.profileImageUrl ? <img src={surgeon.profileImageUrl} alt="" /> : <span>{initials(surgeon.name)}</span>}
      </div>
      <div className="surgeon-map__popover-copy">
        <p className="surgeon-map__eyebrow">Community score · {Math.round(surgeon.communityScore)}</p>
        <h3>{surgeon.name}</h3>
        <p className="surgeon-map__location">{surgeon.locationLabel}</p>
      </div>
      <button type="button" onClick={() => onViewProfile(surgeon.slug)}>
        View profile <span aria-hidden="true">→</span>
      </button>
    </aside>
  );
}

export function SurgeonMap({ surgeons, selectedSurgeonId, onSelectSurgeon, onViewProfile }: SurgeonMapProps) {
  const apiKey = (import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string | undefined)?.trim();
  const [loadState, setLoadState] = useState<MapLoadState>(apiKey ? "loading" : "error");
  const canvasRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<google.maps.Map | null>(null);
  const markersRef = useRef(new Map<string, google.maps.Marker>());
  const selectRef = useRef(onSelectSurgeon);
  selectRef.current = onSelectSurgeon;

  const selectedSurgeon = useMemo(
    () => surgeons.find((surgeon) => surgeon.id === selectedSurgeonId) ?? null,
    [selectedSurgeonId, surgeons],
  );

  useEffect(() => {
    if (!apiKey) return;
    let active = true;
    void loadGoogleMaps(apiKey).then(
      () => active && setLoadState("ready"),
      () => active && setLoadState("error"),
    );
    return () => {
      active = false;
    };
  }, [apiKey]);

  useEffect(() => {
    if (loadState !== "ready" || !canvasRef.current) return;

    const map =
      mapRef.current ??
      new google.maps.Map(canvasRef.current, {
        center: MIAMI_CENTER,
        zoom: 11,
        mapId: "DEMO_MAP_ID",
        disableDefaultUI: true,
        zoomControl: true,
        clickableIcons: false,
        gestureHandling: "cooperative",
      });
    mapRef.current = map;

    for (const marker of markersRef.current.values()) marker.setMap(null);
    markersRef.current.clear();

    const bounds = new google.maps.LatLngBounds();
    for (const surgeon of surgeons) {
      const selected = surgeon.id === selectedSurgeonId;
      const marker = new google.maps.Marker({
        map,
        position: { lat: surgeon.latitude, lng: surgeon.longitude },
        title: `${surgeon.name}, community score ${Math.round(surgeon.communityScore)}`,
        icon: markerIcon(selected),
        label: scoreLabel(surgeon.communityScore),
        zIndex: selected ? 20 : 10,
      });
      marker.addListener("click", () => selectRef.current(surgeon.id));
      markersRef.current.set(surgeon.id, marker);
      bounds.extend({ lat: surgeon.latitude, lng: surgeon.longitude });
    }

    if (surgeons.length === 1) {
      map.setCenter({ lat: surgeons[0].latitude, lng: surgeons[0].longitude });
      map.setZoom(12);
    } else if (surgeons.length > 1) {
      map.fitBounds(bounds, { top: 76, right: 52, bottom: 76, left: 52 });
    }

    return () => {
      for (const marker of markersRef.current.values()) marker.setMap(null);
      markersRef.current.clear();
    };
  }, [loadState, surgeons]);

  useEffect(() => {
    if (loadState !== "ready") return;
    for (const [id, marker] of markersRef.current) {
      const selected = id === selectedSurgeonId;
      marker.setIcon(markerIcon(selected));
      marker.setZIndex(selected ? 20 : 10);
    }
  }, [loadState, selectedSurgeonId]);

  const useFallback = loadState === "error";

  return (
    <section className="surgeon-map" aria-label="Surgeon locations">
      <div className="surgeon-map__topline">
        <div>
          <span className="surgeon-map__kicker">Explore locally</span>
          <strong>Miami surgeons</strong>
        </div>
        <span className="surgeon-map__count">{surgeons.length} practices</span>
      </div>

      <div className={`surgeon-map__stage${useFallback ? " is-fallback" : ""}`}>
        {loadState === "loading" && (
          <div className="surgeon-map__loading" role="status">
            <span />
            <p>Plotting community favorites…</p>
          </div>
        )}

        {loadState === "ready" && <div ref={canvasRef} className="surgeon-map__google" aria-label="Interactive map of Miami surgeons" />}

        {useFallback && (
          <div className="surgeon-map__fallback" aria-label="Map-style view of Miami surgeons">
            <span className="surgeon-map__water-label" aria-hidden="true">Biscayne Bay</span>
            <span className="surgeon-map__place is-north" aria-hidden="true">MIAMI BEACH</span>
            <span className="surgeon-map__place is-center" aria-hidden="true">DOWNTOWN</span>
            <span className="surgeon-map__place is-south" aria-hidden="true">CORAL GABLES</span>
            <span className="surgeon-map__road is-one" aria-hidden="true" />
            <span className="surgeon-map__road is-two" aria-hidden="true" />
            <span className="surgeon-map__road is-three" aria-hidden="true" />
            {surgeons.map((surgeon) => {
              const position = deterministicPosition(surgeon);
              const selected = surgeon.id === selectedSurgeonId;
              return (
                <button
                  key={surgeon.id}
                  className={`surgeon-map__fallback-marker${selected ? " is-selected" : ""}`}
                  style={{ left: `${position.left}%`, top: `${position.top}%` }}
                  type="button"
                  aria-pressed={selected}
                  aria-label={`${surgeon.name}, community score ${Math.round(surgeon.communityScore)}`}
                  onClick={() => onSelectSurgeon(surgeon.id)}
                >
                  <span>{Math.round(surgeon.communityScore)}</span>
                </button>
              );
            })}
          </div>
        )}

        {selectedSurgeon && <SurgeonPopover surgeon={selectedSurgeon} onViewProfile={onViewProfile} />}

        <div className="surgeon-map__legend" aria-hidden="true">
          <span /> Community score
        </div>
      </div>
    </section>
  );
}
