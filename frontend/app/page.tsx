"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type Detection = {
  class: string;
  confidence: number;
  objectness: number;
  temperature_c: number;
  laplacian_variance: number;
  bbox: number[];
  merged_parts: number;
};

type AnalyzeResult = {
  ok: boolean;
  mode?: string;
  count?: number;
  elapsed_ms?: number;
  thresholds?: {
    confidence?: number | null;
    laplacian?: number;
    lap_bypass_confidence?: number | null;
  };
  detections?: Detection[];
  images?: {
    annotated_primary?: string;
    annotated_thermal?: string;
  };
  error?: string;
  detail?: string;
};

function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }

    const nextUrl = URL.createObjectURL(file);
    setUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  return url;
}

function UploadCard({
  title,
  hint,
  file,
  previewUrl,
  onChange,
  onClear
}: {
  title: string;
  hint: string;
  file: File | null;
  previewUrl: string | null;
  onChange: (file: File | null) => void;
  onClear: () => void;
}) {
  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    onChange(event.target.files?.[0] ?? null);
  }

  return (
    <section className="upload-card">
      <div className="upload-card__copy">
        <p className="eyebrow">{title}</p>
        <p>{hint}</p>
      </div>
      <label className={`drop-zone ${previewUrl ? "drop-zone--has-preview" : ""}`}>
        {previewUrl ? (
          <img src={previewUrl} alt={`${title} preview`} />
        ) : (
          <span>Choose image</span>
        )}
        <input type="file" accept="image/*" onChange={handleChange} />
      </label>
      <div className="file-row">
        <span>{file ? file.name : "No file selected"}</span>
        {file ? (
          <button type="button" className="ghost-button" onClick={onClear}>
            Clear
          </button>
        ) : null}
      </div>
    </section>
  );
}

export default function Home() {
  const [thermalFile, setThermalFile] = useState<File | null>(null);
  const [rgbFile, setRgbFile] = useState<File | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [lapThresh, setLapThresh] = useState(80);
  const [customConf, setCustomConf] = useState(false);
  const [confThresh, setConfThresh] = useState(0.2);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);

  const thermalPreview = useObjectUrl(thermalFile);
  const rgbPreview = useObjectUrl(rgbFile);

  const inputMode = useMemo(() => {
    if (rgbFile && thermalFile) return "Paired RGB + thermal";
    if (thermalFile) return "Thermal only";
    if (rgbFile) return "RGB only";
    return "Waiting for an image";
  }, [rgbFile, thermalFile]);

  async function analyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setResult(null);

    if (!thermalFile && !rgbFile) {
      setError("Upload at least one RGB or thermal image.");
      return;
    }

    const formData = new FormData();
    if (thermalFile) formData.append("thermal_image", thermalFile);
    if (rgbFile) formData.append("rgb_image", rgbFile);
    formData.append("lap_thresh", String(lapThresh));
    if (customConf) formData.append("conf_thresh", String(confThresh));

    setLoading(true);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });
      const payload = (await response.json()) as AnalyzeResult;
      if (!response.ok || payload.ok === false) {
        setError(payload.error || payload.detail || "Inference failed.");
        return;
      }
      setResult(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not analyze image.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">MicroGhost web inference</p>
          <h1>Thermal intrusion detection, ready for field images.</h1>
          <p className="hero-copy">
            Upload paired RGB and thermal images, or provide a single thermal or RGB frame.
            The analysis runs on the hosted Vercel backend.
          </p>
        </div>
        <div className="status-panel">
          <span>Input mode</span>
          <strong>{inputMode}</strong>
        </div>
      </section>

      <form className="workspace" onSubmit={analyze}>
        <div className="upload-grid">
          <UploadCard
            title="Thermal image"
            hint="Recommended for best detection. Single thermal images are supported."
            file={thermalFile}
            previewUrl={thermalPreview}
            onChange={setThermalFile}
            onClear={() => setThermalFile(null)}
          />
          <UploadCard
            title="RGB image"
            hint="Optional paired visible image. Helps reject smooth hot objects."
            file={rgbFile}
            previewUrl={rgbPreview}
            onChange={setRgbFile}
            onClear={() => setRgbFile(null)}
          />
        </div>

        <section className="control-strip">
          <button type="submit" className="primary-button" disabled={loading}>
            {loading ? "Analyzing..." : "Analyze"}
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => setAdvancedOpen((open) => !open)}
          >
            Advanced tuning
          </button>
          <p>
            Laplacian filtering removes flat hot surfaces. Single-image confidence uses
            a more sensitive default unless you override it.
          </p>
        </section>

        {advancedOpen ? (
          <section className="advanced-panel">
            <label className="slider-row">
              <span>Lap threshold</span>
              <input
                type="range"
                min="0"
                max="220"
                step="5"
                value={lapThresh}
                onChange={(event) => setLapThresh(Number(event.target.value))}
              />
              <strong>{lapThresh}</strong>
            </label>
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={customConf}
                onChange={(event) => setCustomConf(event.target.checked)}
              />
              <span>Override confidence threshold</span>
            </label>
            <label className={`slider-row ${customConf ? "" : "slider-row--disabled"}`}>
              <span>Confidence</span>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.01"
                value={confThresh}
                disabled={!customConf}
                onChange={(event) => setConfThresh(Number(event.target.value))}
              />
              <strong>{confThresh.toFixed(2)}</strong>
            </label>
          </section>
        ) : null}
      </form>

      {error ? <section className="error-panel">{error}</section> : null}

      {result ? (
        <section className="results">
          <div className="result-summary">
            <div>
              <span>Detections</span>
              <strong>{result.count ?? 0}</strong>
            </div>
            <div>
              <span>Mode</span>
              <strong>{result.mode?.replace("_", " ") ?? "unknown"}</strong>
            </div>
            <div>
              <span>Runtime</span>
              <strong>{result.elapsed_ms ?? 0} ms</strong>
            </div>
            <div>
              <span>Lap threshold</span>
              <strong>{result.thresholds?.laplacian ?? lapThresh}</strong>
            </div>
          </div>

          <div className="result-images">
            {result.images?.annotated_primary ? (
              <figure>
                <img src={result.images.annotated_primary} alt="Annotated primary result" />
                <figcaption>Primary view</figcaption>
              </figure>
            ) : null}
            {result.images?.annotated_thermal ? (
              <figure>
                <img src={result.images.annotated_thermal} alt="Annotated thermal result" />
                <figcaption>Thermal view</figcaption>
              </figure>
            ) : null}
          </div>

          <div className="detections-table">
            <div className="detections-table__head">
              <span>Class</span>
              <span>Conf</span>
              <span>Lap var</span>
              <span>Box</span>
            </div>
            {(result.detections ?? []).map((det, index) => (
              <div className="detections-table__row" key={`${det.class}-${index}`}>
                <span>{det.class}</span>
                <span>{det.confidence.toFixed(2)}</span>
                <span>{det.laplacian_variance.toFixed(1)}</span>
                <span>{det.bbox.map((v) => v.toFixed(2)).join(", ")}</span>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </main>
  );
}
