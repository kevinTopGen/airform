import { useEffect, useRef, useState, type ChangeEvent, type PointerEvent } from "react";
import inputFaceUrl from "../../../fixtures/input_front.jpg";
import { demoSurgeons, type SurgeonProfile } from "../data/surgeons";
import { getSignaturePrior, type SignaturePrior } from "./signaturePriors";

interface BrowserPreviewProps {
  surgeon: SurgeonProfile;
  navigate: (path: string) => void;
}

interface Point {
  x: number;
  y: number;
}

interface LoadedImage {
  element: HTMLImageElement;
  width: number;
  height: number;
}

const DEFAULT_TARGET: Point = { x: 0.5, y: 0.49 };
const MAX_IMAGE_EDGE = 1200;

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.min(maximum, Math.max(minimum, value));

export function warpNoseRegion(
  source: ImageData,
  target: Point,
  regionSize: number,
  strength: number,
  prior: SignaturePrior,
): ImageData {
  const { width, height, data } = source;
  const output = new ImageData(new Uint8ClampedArray(data), width, height);
  const centerX = target.x * width;
  const centerY = target.y * height;
  const radiusX = Math.max(24, width * regionSize * 0.5);
  const radiusY = Math.max(36, radiusX * 1.55);
  const intensity = clamp(strength, 0, 1);
  const visualGain = 1.45;
  const minX = Math.max(0, Math.floor(centerX - radiusX));
  const maxX = Math.min(width - 1, Math.ceil(centerX + radiusX));
  const minY = Math.max(0, Math.floor(centerY - radiusY));
  const maxY = Math.min(height - 1, Math.ceil(centerY + radiusY));

  for (let y = minY; y <= maxY; y += 1) {
    const normalizedY = (y - centerY) / radiusY;
    const bridgeWeight = Math.exp(-Math.pow((normalizedY + 0.43) / 0.42, 2));
    const tipWeight = Math.exp(-Math.pow((normalizedY - 0.13) / 0.34, 2));
    const alarWeight = Math.exp(-Math.pow((normalizedY - 0.48) / 0.35, 2));
    const weightTotal = bridgeWeight + tipWeight + alarWeight;
    const widthDelta =
      (prior.delta.bridgeWidth * bridgeWeight +
        prior.delta.tipWidth * tipWeight +
        prior.delta.alarWidth * alarWeight) /
      weightTotal;
    const horizontalScale = clamp(1 + widthDelta * intensity * visualGain, 0.62, 1.12);
    const verticalShift = prior.delta.nasalLength * intensity * height * 0.46 * (0.35 + alarWeight * 0.65);

    for (let x = minX; x <= maxX; x += 1) {
      const normalizedX = (x - centerX) / radiusX;
      const distance = normalizedX * normalizedX + normalizedY * normalizedY;
      if (distance >= 1) continue;

      const feather = Math.pow(1 - distance, 0.58) * intensity;
      const sampleX = clamp(Math.round(centerX + (x - centerX) / horizontalScale), 0, width - 1);
      const sampleY = clamp(Math.round(y - verticalShift), 0, height - 1);
      const sourceIndex = (sampleY * width + sampleX) * 4;
      const destinationIndex = (y * width + x) * 4;

      for (let channel = 0; channel < 3; channel += 1) {
        output.data[destinationIndex + channel] = Math.round(
          data[destinationIndex + channel] * (1 - feather) + data[sourceIndex + channel] * feather,
        );
      }
    }
  }

  return output;
}

export function BrowserPreview({ surgeon, navigate }: BrowserPreviewProps) {
  const prior = getSignaturePrior(surgeon.signatureId);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const uploadUrlRef = useRef<string | null>(null);
  const [sourceUrl, setSourceUrl] = useState(inputFaceUrl);
  const [sourceName, setSourceName] = useState("Bundled demo portrait");
  const [loadedImage, setLoadedImage] = useState<LoadedImage | null>(null);
  const [target, setTarget] = useState<Point>(DEFAULT_TARGET);
  const [regionSize, setRegionSize] = useState(0.2);
  const [strength, setStrength] = useState(0.9);
  const [reveal, setReveal] = useState(52);
  const [previewReady, setPreviewReady] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [message, setMessage] = useState("Click the nose to reposition the target, then generate.");

  useEffect(() => {
    const image = new Image();
    image.onload = () => {
      const scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(image.naturalWidth, image.naturalHeight));
      setLoadedImage({
        element: image,
        width: Math.round(image.naturalWidth * scale),
        height: Math.round(image.naturalHeight * scale),
      });
      setPreviewReady(false);
      setMessage("Click the nose to reposition the target, then generate.");
    };
    image.onerror = () => {
      setLoadedImage(null);
      setMessage("That image could not be opened. Try a JPG, PNG, or WebP portrait.");
    };
    image.src = sourceUrl;
    return () => {
      image.onload = null;
      image.onerror = null;
    };
  }, [sourceUrl]);

  useEffect(() => {
    setPreviewReady(false);
    setMessage(`${prior.name} is selected. Generate to apply its signature prior.`);
  }, [prior.id, prior.name]);

  useEffect(
    () => () => {
      if (uploadUrlRef.current) URL.revokeObjectURL(uploadUrlRef.current);
    },
    [],
  );

  const selectFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setMessage("Choose an image file to continue.");
      return;
    }
    if (uploadUrlRef.current) URL.revokeObjectURL(uploadUrlRef.current);
    const nextUrl = URL.createObjectURL(file);
    uploadUrlRef.current = nextUrl;
    setSourceUrl(nextUrl);
    setSourceName(file.name);
    setTarget(DEFAULT_TARGET);
    event.target.value = "";
  };

  const useBundledPortrait = () => {
    if (uploadUrlRef.current) URL.revokeObjectURL(uploadUrlRef.current);
    uploadUrlRef.current = null;
    setSourceUrl(inputFaceUrl);
    setSourceName("Bundled demo portrait");
    setTarget(DEFAULT_TARGET);
  };

  const repositionTarget = (event: PointerEvent<HTMLDivElement>) => {
    if (previewReady) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    setTarget({
      x: clamp((event.clientX - bounds.left) / bounds.width, 0.08, 0.92),
      y: clamp((event.clientY - bounds.top) / bounds.height, 0.1, 0.9),
    });
    setMessage("Target repositioned. Generate when the oval covers the nose.");
  };

  const generatePreview = () => {
    if (!loadedImage || !canvasRef.current) return;
    setProcessing(true);
    setMessage("Applying a nose-localized geometric warp…");

    window.requestAnimationFrame(() => {
      const sourceCanvas = document.createElement("canvas");
      sourceCanvas.width = loadedImage.width;
      sourceCanvas.height = loadedImage.height;
      const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
      const outputContext = canvasRef.current?.getContext("2d");
      if (!sourceContext || !outputContext || !canvasRef.current) {
        setProcessing(false);
        setMessage("Canvas preview is unavailable in this browser.");
        return;
      }

      sourceContext.drawImage(loadedImage.element, 0, 0, loadedImage.width, loadedImage.height);
      const source = sourceContext.getImageData(0, 0, loadedImage.width, loadedImage.height);
      const output = warpNoseRegion(source, target, regionSize, strength, prior);
      canvasRef.current.width = loadedImage.width;
      canvasRef.current.height = loadedImage.height;
      outputContext.putImageData(output, 0, 0);
      setPreviewReady(true);
      setProcessing(false);
      setReveal(52);
      setMessage("Preview generated locally in your browser. Drag the divider to compare.");
    });
  };

  return (
    <main className="browser-preview">
      <header className="browser-preview__header">
        <button className="back-link" type="button" onClick={() => navigate(`/surgeons/${surgeon.slug}`)}>
          <span aria-hidden="true">←</span> {surgeon.name}
        </button>
        <div className="browser-preview__status"><span /> Hackathon visualization · local browser processing</div>
        <button className="browser-preview__rank" type="button" onClick={() => navigate("/tournament")}>Community ranking →</button>
      </header>

      <section className="browser-preview__workspace" aria-labelledby="preview-title">
        <div className="browser-preview__intro">
          <span className="browser-preview__step">Visual study / 01</span>
          <h1 id="preview-title">Try the shape,<br />keep your face.</h1>
          <p>This demo uses a localized canvas warp inspired by Airform’s geometric pipeline. It is an aesthetic visualization—not a predicted surgical result.</p>
        </div>

        <div className="browser-preview__image-column">
          <div
            className={`browser-preview__frame${previewReady ? " is-ready" : ""}`}
            style={loadedImage ? { aspectRatio: `${loadedImage.width} / ${loadedImage.height}` } : undefined}
            onPointerDown={repositionTarget}
          >
            {loadedImage ? <img src={sourceUrl} alt="Portrait selected for visualization" draggable={false} /> : <div className="browser-preview__image-error">Portrait unavailable</div>}
            <canvas
              ref={canvasRef}
              className="browser-preview__result"
              style={{ clipPath: `inset(0 0 0 ${reveal}%)` }}
              aria-label="Generated Airform preview"
            />
            {!previewReady && loadedImage ? (
              <span
                className="browser-preview__target"
                style={{ left: `${target.x * 100}%`, top: `${target.y * 100}%`, width: `${regionSize * 100}%` }}
                aria-hidden="true"
              ><span /></span>
            ) : null}
            {previewReady ? <><span className="browser-preview__before-label">Original</span><span className="browser-preview__after-label">Visualization</span><span className="browser-preview__divider" style={{ left: `${reveal}%` }} /></> : null}
            {processing ? <div className="browser-preview__processing" role="status"><span /> Rendering</div> : null}
          </div>
          {previewReady ? (
            <label className="browser-preview__compare-slider">
              <span>Original</span>
              <input type="range" min="8" max="92" value={reveal} onChange={(event) => setReveal(Number(event.target.value))} aria-label="Before and after comparison divider" />
              <span>Preview</span>
            </label>
          ) : null}
          <p className="browser-preview__message" role="status">{message}</p>
        </div>

        <aside className="browser-preview__controls" aria-label="Preview controls">
          <div className="browser-preview__control-group">
            <span className="browser-preview__control-number">01</span>
            <div><label htmlFor="preview-surgeon">Aesthetic prior</label><select id="preview-surgeon" value={surgeon.slug} onChange={(event) => navigate(`/preview/${event.target.value}`)}>{demoSurgeons.map((item) => <option key={item.id} value={item.slug}>{item.name}</option>)}</select><strong>{prior.name}</strong><p>{prior.tagline}</p></div>
          </div>
          <div className="browser-preview__control-group">
            <span className="browser-preview__control-number">02</span>
            <div><span className="browser-preview__label">Portrait</span><strong className="browser-preview__filename">{sourceName}</strong><div className="browser-preview__source-actions"><label className="browser-preview__upload">Upload portrait<input type="file" accept="image/jpeg,image/png,image/webp" onChange={selectFile} /></label><button type="button" onClick={useBundledPortrait}>Use bundled face</button></div></div>
          </div>
          <div className="browser-preview__control-group">
            <span className="browser-preview__control-number">03</span>
            <div className="browser-preview__ranges"><label>Effect strength <output>{Math.round(strength * 100)}%</output><input type="range" min="35" max="100" value={strength * 100} onChange={(event) => { setStrength(Number(event.target.value) / 100); setPreviewReady(false); }} /></label><label>Target size <output>{Math.round(regionSize * 100)}%</output><input type="range" min="14" max="28" value={regionSize * 100} onChange={(event) => { setRegionSize(Number(event.target.value) / 100); setPreviewReady(false); }} /></label></div>
          </div>
          <button className="browser-preview__generate" type="button" disabled={!loadedImage || processing} onClick={generatePreview}>{processing ? "Applying geometry…" : previewReady ? "Regenerate preview" : "Generate preview"}<span aria-hidden="true">↗</span></button>
          <p className="browser-preview__fineprint">Runs entirely in this tab. Uploaded images are not sent or stored.</p>
        </aside>
      </section>
    </main>
  );
}

