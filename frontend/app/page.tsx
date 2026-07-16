"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type Detection = {
  class: string;
  confidence: number;
  objectness: number;
  thermal_score: number;
  laplacian_variance: number;
  bbox: number[];
  merged_parts: number;
};

type PerformanceMetrics = {
  wall_time_ms?: number;
  process_cpu_time_ms?: number;
  estimated_cpu_load_percent?: number | null;
  cpu_percent?: number | null;
  cpu_frequency_mhz?: number | null;
  process_memory_gb?: number | null;
  system_memory_percent?: number | null;
  system_memory_used_gb?: number | null;
  system_memory_total_gb?: number | null;
  thread_count?: number | null;
  cpu_count?: number | null;
  model_runtime?: string;
  max_upload_mb?: number;
  max_image_side_px?: number;
};

type AnalyzeResult = {
  ok: boolean;
  mode?: string;
  count?: number;
  elapsed_ms?: number;
  thresholds?: {
    confidence?: number | null;
    laplacian?: number;
    effective_laplacian?: number;
    lap_bypass_confidence?: number | null;
    merge_boxes?: boolean;
  };
  performance?: PerformanceMetrics;
  detections?: Detection[];
  images?: {
    annotated_primary?: string;
    annotated_thermal?: string;
  };
  error?: string;
  detail?: string;
};

type RunLog = {
  id: number;
  label: string;
  detections: number;
  runtime: string;
  cpu: string;
  memory: string;
};

const DEFAULT_LAP_THRESHOLD = 80;

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

function formatNumber(value: number | null | undefined, suffix = "", digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${value.toFixed(digits)}${suffix}`;
}

function modeLabel(mode?: string) {
  if (!mode) return "Ready";
  return mode.replace(/_/g, " ");
}

function getClientMemoryLabel() {
  const memory = (performance as Performance & {
    memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number };
  }).memory;

  if (!memory) return "Browser memory n/a";
  const usedMb = memory.usedJSHeapSize / (1024 * 1024);
  const limitMb = memory.jsHeapSizeLimit / (1024 * 1024);
  return `${usedMb.toFixed(0)} MB / ${limitMb.toFixed(0)} MB JS heap`;
}

function downloadDataUrl(dataUrl: string, filename: string) {
  const anchor = document.createElement("a");
  anchor.href = dataUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

function LogoMark() {
  const [useFallback, setUseFallback] = useState(false);

  return (
    <div className="brand-mark" aria-label="MicroGhost">
      {!useFallback ? (
        <img src="/logo.png" alt="MicroGhost logo" onError={() => setUseFallback(true)} />
      ) : (
        <span>MG</span>
      )}
    </div>
  );
}

function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const saved = window.localStorage.getItem("microghost-theme");
    const nextTheme = saved === "dark" ? "dark" : "light";
    setTheme(nextTheme);
    document.documentElement.dataset.theme = nextTheme;
  }, []);

  function toggleTheme() {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    window.localStorage.setItem("microghost-theme", nextTheme);
    document.documentElement.dataset.theme = nextTheme;
  }

  return (
    <button type="button" className="icon-button theme-toggle" onClick={toggleTheme}>
      <span aria-hidden="true">{theme === "dark" ? "L" : "D"}</span>
      <span>{theme === "dark" ? "Light" : "Dark"}</span>
    </button>
  );
}

function UploadCard({
  title,
  hint,
  file,
  previewUrl,
  accent,
  onChange,
  onClear
}: {
  title: string;
  hint: string;
  file: File | null;
  previewUrl: string | null;
  accent: "thermal" | "rgb";
  onChange: (file: File | null) => void;
  onClear: () => void;
}) {
  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    onChange(event.target.files?.[0] ?? null);
  }

  return (
    <section className={`upload-card upload-card--${accent}`}>
      <div className="upload-card__copy">
        <p className="eyebrow">{title}</p>
        <p>{hint}</p>
      </div>
      <label className={`drop-zone ${previewUrl ? "drop-zone--has-preview" : ""}`}>
        {previewUrl ? (
          <img src={previewUrl} alt={`${title} preview`} />
        ) : (
          <span>
            <strong>Choose image</strong>
            <small>PNG, JPG, or thermal frame</small>
          </span>
        )}
        <input type="file" accept="image/*" onChange={handleChange} />
      </label>
      <div className="file-row">
        <span title={file?.name}>{file ? file.name : "No file selected"}</span>
        {file ? (
          <button type="button" className="ghost-button" onClick={onClear}>
            Clear
          </button>
        ) : null}
      </div>
    </section>
  );
}

function ProgressPanel({
  loading,
  progress,
  elapsedSeconds,
  clientStatus
}: {
  loading: boolean;
  progress: number;
  elapsedSeconds: number;
  clientStatus: string;
}) {
  return (
    <section className={`progress-panel ${loading ? "progress-panel--active" : ""}`}>
      <div>
        <span>Inference stream</span>
        <strong>{loading ? "Analyzing frame" : "Standing by"}</strong>
      </div>
      <div className="progress-track" aria-label="Inference progress">
        <span style={{ width: `${progress}%` }} />
      </div>
      <p>
        {loading
          ? `${elapsedSeconds.toFixed(1)}s elapsed - ${clientStatus}`
          : "Upload a frame and run detection to populate live metrics."}
      </p>
    </section>
  );
}

function MetricCard({ label, value, tone }: { label: string; value: string; tone?: "hot" | "cool" }) {
  return (
    <div className={`metric-card ${tone ? `metric-card--${tone}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PerformanceGrid({ result }: { result: AnalyzeResult | null }) {
  const perf = result?.performance;

  return (
    <section className="metrics-grid">
      <MetricCard label="Wall time" value={perf ? `${perf.wall_time_ms ?? result?.elapsed_ms ?? 0} ms` : "n/a"} tone="cool" />
      <MetricCard
        label="CPU load"
        value={formatNumber(perf?.estimated_cpu_load_percent ?? perf?.cpu_percent, "%")}
        tone="hot"
      />
      <MetricCard label="CPU clock" value={formatNumber(perf?.cpu_frequency_mhz, " MHz", 0)} />
      <MetricCard label="Process RAM" value={formatNumber(perf?.process_memory_gb, " GB", 3)} />
      <MetricCard
        label="System RAM"
        value={
          perf?.system_memory_used_gb && perf?.system_memory_total_gb
            ? `${perf.system_memory_used_gb.toFixed(2)} / ${perf.system_memory_total_gb.toFixed(2)} GB`
            : "n/a"
        }
      />
      <MetricCard label="Threads / cores" value={`${perf?.thread_count ?? "n/a"} / ${perf?.cpu_count ?? "n/a"}`} />
    </section>
  );
}

export default function Home() {
  const [thermalFile, setThermalFile] = useState<File | null>(null);
  const [rgbFile, setRgbFile] = useState<File | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [lapThresh, setLapThresh] = useState(DEFAULT_LAP_THRESHOLD);
  const [customConf, setCustomConf] = useState(false);
  const [confThresh, setConfThresh] = useState(0.2);
  const [mergeBoxes, setMergeBoxes] = useState(false);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [clientStatus, setClientStatus] = useState("Browser metrics ready");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [runLogs, setRunLogs] = useState<RunLog[]>([]);

  const thermalPreview = useObjectUrl(thermalFile);
  const rgbPreview = useObjectUrl(rgbFile);

  const inputMode = useMemo(() => {
    if (rgbFile && thermalFile) return "Paired RGB + thermal";
    if (thermalFile) return "Thermal only";
    if (rgbFile) return "RGB only";
    return "Waiting for an image";
  }, [rgbFile, thermalFile]);

  useEffect(() => {
    if (!loading) return;

    const started = performance.now();
    setProgress(6);
    setElapsedSeconds(0);
    setClientStatus(`${navigator.hardwareConcurrency ?? "n/a"} logical cores - ${getClientMemoryLabel()}`);

    const timer = window.setInterval(() => {
      const elapsed = (performance.now() - started) / 1000;
      setElapsedSeconds(elapsed);
      setProgress(Math.min(94, 12 + (1 - Math.exp(-elapsed / 3.2)) * 82));
      setClientStatus(`${navigator.hardwareConcurrency ?? "n/a"} logical cores - ${getClientMemoryLabel()}`);
    }, 220);

    return () => window.clearInterval(timer);
  }, [loading]);

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
    formData.append("merge_boxes", String(mergeBoxes));
    if (customConf) formData.append("conf_thresh", String(confThresh));

    setLoading(true);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });
      const payload = (await response.json()) as AnalyzeResult;
      setProgress(100);
      if (!response.ok || payload.ok === false) {
        setError(payload.error || payload.detail || "Inference failed.");
        return;
      }
      setResult(payload);
      setRunLogs((logs) => [
        {
          id: Date.now(),
          label: modeLabel(payload.mode),
          detections: payload.count ?? 0,
          runtime: `${payload.performance?.wall_time_ms ?? payload.elapsed_ms ?? 0} ms`,
          cpu: formatNumber(
            payload.performance?.estimated_cpu_load_percent ?? payload.performance?.cpu_percent,
            "%"
          ),
          memory: formatNumber(payload.performance?.process_memory_gb, " GB", 3)
        },
        ...logs
      ].slice(0, 5));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not analyze image.");
    } finally {
      setLoading(false);
    }
  }

  const primaryImage = result?.images?.annotated_primary;
  const thermalImage = result?.images?.annotated_thermal;

  return (
    <main className="page-shell">
      <nav className="topbar">
        <div className="brand">
          <LogoMark />
          <div>
            <strong>MicroGhost</strong>
            <span>Thermal inference console</span>
          </div>
        </div>
        <ThemeToggle />
      </nav>

      <section className="hero">
        <div className="hero__content">
          <p className="eyebrow">RGB + thermal detection</p>
          <h1>One-page intrusion detection with live inference telemetry.</h1>
          <p className="hero-copy">
            Upload RGB, thermal, or paired frames. The demo runs ONNX inference,
            filters detections with Laplacian texture checks, and returns annotated images.
          </p>
        </div>
        <div className="status-panel">
          <span>Input mode</span>
          <strong>{inputMode}</strong>
          <small>Lap threshold {lapThresh} - {customConf ? `conf ${confThresh.toFixed(2)}` : "default confidence"}</small>
        </div>
      </section>

      <form className="workspace" onSubmit={analyze}>
        <div className="upload-grid">
          <UploadCard
            title="Thermal image"
            hint="Best signal for camouflaged or low-light targets. Uses Laplacian threshold 80 by default."
            file={thermalFile}
            previewUrl={thermalPreview}
            accent="thermal"
            onChange={setThermalFile}
            onClear={() => setThermalFile(null)}
          />
          <UploadCard
            title="RGB image"
            hint="Adds visible-context confirmation and supports RGB-only testing when thermal is unavailable."
            file={rgbFile}
            previewUrl={rgbPreview}
            accent="rgb"
            onChange={setRgbFile}
            onClear={() => setRgbFile(null)}
          />
        </div>

        <section className="control-strip">
          <button type="submit" className="primary-button" disabled={loading}>
            {loading ? "Analyzing..." : "Run inference"}
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => setAdvancedOpen((open) => !open)}
          >
            {advancedOpen ? "Hide tuning" : "Tune thresholds"}
          </button>
          <label className="switch-row">
            <input
              type="checkbox"
              checked={mergeBoxes}
              onChange={(event) => setMergeBoxes(event.target.checked)}
            />
            <span>Merge related boxes</span>
          </label>
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
            <label className="switch-row switch-row--panel">
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
                min="0.05"
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

      <ProgressPanel
        loading={loading}
        progress={progress}
        elapsedSeconds={elapsedSeconds}
        clientStatus={clientStatus}
      />

      {error ? <section className="error-panel">{error}</section> : null}

      <section className="results">
        <div className="result-summary">
          <MetricCard label="Detections" value={String(result?.count ?? 0)} tone="hot" />
          <MetricCard label="Mode" value={modeLabel(result?.mode)} />
          <MetricCard
            label="Effective lap"
            value={String(result?.thresholds?.effective_laplacian ?? lapThresh)}
          />
          <MetricCard
            label="Confidence"
            value={
              result?.thresholds?.confidence === null || result?.thresholds?.confidence === undefined
                ? "default"
                : result.thresholds.confidence.toFixed(2)
            }
          />
        </div>

        <PerformanceGrid result={result} />

        <div className="result-images">
          <figure className={!primaryImage ? "figure--empty" : ""}>
            {primaryImage ? (
              <img src={primaryImage} alt="Annotated primary result" />
            ) : (
              <div className="empty-result">Primary detection image</div>
            )}
            <figcaption>
              <span>Primary view</span>
              {primaryImage ? (
                <button
                  type="button"
                  className="download-button"
                  onClick={() => downloadDataUrl(primaryImage, "microghost-primary-detection.jpg")}
                >
                  Download
                </button>
              ) : null}
            </figcaption>
          </figure>
          <figure className={!thermalImage ? "figure--empty" : ""}>
            {thermalImage ? (
              <img src={thermalImage} alt="Annotated thermal result" />
            ) : (
              <div className="empty-result">Thermal detection image</div>
            )}
            <figcaption>
              <span>Thermal view</span>
              {thermalImage ? (
                <button
                  type="button"
                  className="download-button"
                  onClick={() => downloadDataUrl(thermalImage, "microghost-thermal-detection.jpg")}
                >
                  Download
                </button>
              ) : null}
            </figcaption>
          </figure>
        </div>

        <div className="detections-table">
          <div className="detections-table__head">
            <span>Class</span>
            <span>Conf</span>
            <span>Lap var</span>
            <span>Box</span>
          </div>
          {(result?.detections ?? []).length > 0 ? (
            result?.detections?.map((det, index) => (
              <div className="detections-table__row" key={`${det.class}-${index}`}>
                <span>{det.class}</span>
                <span>{det.confidence.toFixed(2)}</span>
                <span>{det.laplacian_variance.toFixed(1)}</span>
                <span>{det.bbox.map((value) => value.toFixed(2)).join(", ")}</span>
              </div>
            ))
          ) : (
            <div className="detections-table__empty">No detections logged yet.</div>
          )}
        </div>

        <aside className="run-log">
          <div className="run-log__title">
            <span>Recent inference log</span>
            <strong>{runLogs.length}</strong>
          </div>
          {runLogs.length > 0 ? (
            runLogs.map((log) => (
              <div className="run-log__row" key={log.id}>
                <span>{log.label}</span>
                <span>{log.detections} det</span>
                <span>{log.runtime}</span>
                <span>{log.cpu}</span>
                <span>{log.memory}</span>
              </div>
            ))
          ) : (
            <p>Inference history appears after the first run.</p>
          )}
        </aside>
      </section>
    </main>
  );
}
