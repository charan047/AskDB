/* eslint-disable no-unused-vars */
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

/**
 * AskDB Frontend (WOW light UI)
 * - Same functionality
 * - High-contrast "big tech" light theme
 * - No extra dependencies (just React + Tailwind + react-markdown)
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL;

// ---------------- helpers ----------------
function downloadCSVBlob(rows) {
  if (!rows?.length) return null;
  const cols = Object.keys(rows[0]);

  const escape = (v) => {
    const s = v === null || v === undefined ? "" : String(v);
    const needsQuotes = /[",\n]/.test(s);
    const escaped = s.replace(/"/g, '""');
    return needsQuotes ? `"${escaped}"` : escaped;
  };

  const header = cols.join(",");
  const body = rows.map((r) => cols.map((c) => escape(r[c])).join(",")).join("\n");
  const csv = `${header}\n${body}`;
  return new Blob([csv], { type: "text/csv;charset=utf-8;" });
}

function prettySql(sql) {
  if (!sql) return "";
  let s = String(sql).trim().replace(/\s+/g, " ");

  const keywords = [
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP BY",
    "HAVING",
    "ORDER BY",
    "LIMIT",
    "JOIN",
    "LEFT JOIN",
    "RIGHT JOIN",
    "INNER JOIN",
    "OUTER JOIN",
    "UNION",
    "VALUES",
    "INSERT",
    "UPDATE",
    "DELETE",
  ].sort((a, b) => b.length - a.length);

  for (const kw of keywords) {
    const re = new RegExp(`\\b${kw.replace(" ", "\\s+")}\\b`, "gi");
    s = s.replace(re, `\n${kw}`);
  }
  return s.replace(/\n{2,}/g, "\n").trim();
}

function normalizeLine(line) {
  return String(line || "")
    .replace(/^\s*[*\-\u2022]\s*/, "")
    .replace(/\*\*/g, "")
    .trim();
}

function lineSearchMatch(row, q) {
  if (!q) return true;
  const needle = q.toLowerCase();
  for (const k of Object.keys(row || {})) {
    const v = row?.[k];
    if (v === null || v === undefined) continue;
    if (String(v).toLowerCase().includes(needle)) return true;
  }
  return false;
}

function encodeShareUrl({ q, mode }) {
  const url = new URL(window.location.href);
  url.searchParams.set("q", q);
  url.searchParams.set("mode", mode);
  url.searchParams.set("run", "1");
  return url.toString();
}

function loadSavedQueries() {
  try {
    const raw = localStorage.getItem("askdb_saved_queries");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
function saveSavedQueries(items) {
  localStorage.setItem("askdb_saved_queries", JSON.stringify(items));
}

function isFiniteNumber(x) {
  const n = Number(x);
  return Number.isFinite(n);
}

/**
 * Tiny inline chart (no deps).
 * spec: {type:'bar'|'line', x_key, y_key, title}
 */
function Chart({ spec, rows }) {
  if (!spec || !rows?.length) return null;
  const { type, x_key, y_key, title } = spec;
  if (!type || type === "none" || !x_key || !y_key) return null;

  const data = rows
    .map((r) => ({ x: r[x_key], y: r[y_key] }))
    .filter((d) => d.x !== undefined && isFiniteNumber(d.y))
    .slice(0, 12);

  if (data.length < 2) return null;

  const W = 900,
    H = 250,
    pad = 36;
  const ys = data.map((d) => Number(d.y));
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const ySpan = yMax - yMin || 1;

  const xStep = (W - pad * 2) / (data.length - 1);
  const yScale = (v) => H - pad - ((Number(v) - yMin) / ySpan) * (H - pad * 2);

  const axisStroke = "rgba(15, 23, 42, 0.12)";
  const lineStroke = "rgba(79, 70, 229, 0.95)"; // indigo
  const dotFill = "rgba(15, 23, 42, 0.85)";
  const barFill = "rgba(14, 165, 233, 0.75)"; // sky

  if (type === "line") {
    const dPath = data
      .map((d, i) => `${i === 0 ? "M" : "L"} ${pad + i * xStep} ${yScale(d.y)}`)
      .join(" ");

    return (
      <Card tone="soft" className="p-5">
        <div className="flex items-center justify-between">
          <div className="text-sm font-semibold text-slate-900">{title || "Chart"}</div>
          <div className="text-[11px] text-slate-500">
            x: {x_key} • y: {y_key}
          </div>
        </div>
        <div className="mt-3">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-56">
            <path d={`M ${pad} ${H - pad} L ${W - pad} ${H - pad}`} stroke={axisStroke} strokeWidth="2" />
            <path d={`M ${pad} ${pad} L ${pad} ${H - pad}`} stroke={axisStroke} strokeWidth="2" />
            <path d={dPath} fill="none" stroke={lineStroke} strokeWidth="4" strokeLinecap="round" />
            {data.map((d, i) => (
              <circle key={i} cx={pad + i * xStep} cy={yScale(d.y)} r="5" fill={dotFill} />
            ))}
          </svg>
        </div>
      </Card>
    );
  }

  const slot = (W - pad * 2) / data.length;
  const barW = Math.max(18, slot - 14);

  return (
    <Card tone="soft" className="p-5">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-slate-900">{title || "Chart"}</div>
        <div className="text-[11px] text-slate-500">
          x: {x_key} • y: {y_key}
        </div>
      </div>
      <div className="mt-3">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-56">
          <path d={`M ${pad} ${H - pad} L ${W - pad} ${H - pad}`} stroke={axisStroke} strokeWidth="2" />
          <path d={`M ${pad} ${pad} L ${pad} ${H - pad}`} stroke={axisStroke} strokeWidth="2" />
          {data.map((d, i) => {
            const x = pad + i * slot + (slot - barW) / 2;
            const y = yScale(d.y);
            const h = H - pad - y;
            return <rect key={i} x={x} y={y} width={barW} height={h} rx="10" fill={barFill} />;
          })}
        </svg>
      </div>
    </Card>
  );
}

// ---------------- UI primitives ----------------
function Icon({ name, className = "" }) {
  const base = `inline-block ${className}`;
  switch (name) {
    case "spark":
      return (
        <svg className={base} viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M12 2l1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8L12 2z"
            stroke="currentColor"
            strokeWidth="1.8"
          />
        </svg>
      );
    case "db":
      return (
        <svg className={base} viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <ellipse cx="12" cy="6.5" rx="7.5" ry="3.5" stroke="currentColor" strokeWidth="1.8" />
          <path
            d="M4.5 6.5v5.2C4.5 13.9 7.9 15.5 12 15.5s7.5-1.6 7.5-3.8V6.5"
            stroke="currentColor"
            strokeWidth="1.8"
          />
          <path
            d="M4.5 11.7v5.2c0 2.2 3.4 3.8 7.5 3.8s7.5-1.6 7.5-3.8v-5.2"
            stroke="currentColor"
            strokeWidth="1.8"
          />
        </svg>
      );
    case "shield":
      return (
        <svg className={base} viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M12 2l8 4v6c0 5.1-3.4 9.7-8 10-4.6-.3-8-4.9-8-10V6l8-4z"
            stroke="currentColor"
            strokeWidth="1.8"
          />
          <path d="M9 12l2 2 4-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    case "link":
      return (
        <svg className={base} viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M10 13a5 5 0 0 1 0-7l1-1a5 5 0 0 1 7 7l-1 1"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
          />
          <path
            d="M14 11a5 5 0 0 1 0 7l-1 1a5 5 0 0 1-7-7l1-1"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
          />
        </svg>
      );
    case "wand":
      return (
        <svg className={base} viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M4 20l9-9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <path d="M13 11l7-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <path
            d="M14.5 3l.8 2.6L18 6.5l-2.7.9L14.5 10l-.8-2.6L11 6.5l2.7-.9L14.5 3z"
            stroke="currentColor"
            strokeWidth="1.6"
          />
        </svg>
      );
    case "grid":
      return (
        <svg className={base} viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M4 4h7v7H4V4zM13 4h7v7h-7V4zM4 13h7v7H4v-7zM13 13h7v7h-7v-7z" stroke="currentColor" strokeWidth="1.6" />
        </svg>
      );
    default:
      return null;
  }
}

function BrandMark() {
  return (
    <div className="flex items-center gap-3 select-none">
      <div className="relative h-10 w-10 rounded-2xl bg-gradient-to-br from-sky-400 via-indigo-500 to-fuchsia-500 shadow-[0_18px_55px_-22px_rgba(79,70,229,0.55)]">
        <div className="absolute inset-0 rounded-2xl bg-[radial-gradient(circle_at_30%_20%,rgba(255,255,255,0.75),transparent_45%)] opacity-80" />
        <div className="absolute inset-0 rounded-2xl ring-1 ring-white/45" />
      </div>
      <div className="leading-tight">
        <div className="text-[15px] font-semibold tracking-tight text-slate-900">
          AskDB{" "}
          <span className="ml-1 align-middle text-[10px] px-2 py-0.5 rounded-full bg-slate-900/5 border border-slate-900/10 text-slate-700">
            V1.1
          </span>
        </div>
        <div className="text-[11px] text-slate-600">Natural Language → SQL → Insight</div>
      </div>
    </div>
  );
}

function Card({ children, className = "", tone = "default" }) {
  const tones = {
    default: "bg-white/75 border-slate-900/10",
    soft: "bg-white/70 border-slate-900/10",
    plain: "bg-white border-slate-900/10",
  };
  return (
    <div
      className={[
        "rounded-3xl border shadow-[0_20px_60px_-35px_rgba(15,23,42,0.25)]",
        "backdrop-blur-xl",
        tones[tone] || tones.default,
        className,
      ].join(" ")}
    >
      {children}
    </div>
  );
}

function Pill({ children, tone = "default" }) {
  const tones = {
    default: "bg-slate-900/5 border-slate-900/10 text-slate-700",
    good: "bg-emerald-500/10 border-emerald-600/20 text-emerald-900",
    warn: "bg-indigo-600/10 border-indigo-600/20 text-indigo-900",
    info: "bg-sky-500/10 border-sky-600/20 text-sky-900",
  };
  return (
    <span className={["inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] font-medium", tones[tone] || tones.default].join(" ")}>
      {children}
    </span>
  );
}

function Button({ children, variant = "ghost", className = "", ...props }) {
  const base =
    "h-10 px-4 rounded-2xl text-sm font-semibold inline-flex items-center justify-center gap-2 leading-none whitespace-nowrap transition focus:outline-none focus:ring-4 focus:ring-indigo-500/20 hover:-translate-y-0.5 active:translate-y-0 transition-transform";
  const styles = {
    ghost: "border border-slate-900/10 bg-white hover:bg-slate-50 active:bg-slate-100 text-slate-900",
    soft: "border border-slate-900/10 bg-white/70 hover:bg-white active:bg-slate-50 text-slate-900",
    primary: "border border-indigo-600/20 bg-indigo-600 text-white shadow-[0_18px_55px_-30px_rgba(79,70,229,0.45)] hover:bg-indigo-500 active:bg-indigo-700",
    danger: "border border-rose-500/20 bg-rose-500/10 hover:bg-rose-500/15 active:bg-rose-500/20 text-rose-800",
  };
  return (
    <button className={[base, styles[variant] || styles.ghost, className].join(" ")} {...props}>
      {children}
    </button>
  );
}

function ModalShell({ title, onClose, children, maxWidth = "max-w-3xl" }) {
  return (
    <div className="fixed inset-0 z-[90] bg-slate-900/40 backdrop-blur-sm flex items-center justify-center p-4">
      <div className={["w-full", maxWidth].join(" ")}>
        <Card tone="plain" className="p-5">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="text-lg font-semibold tracking-tight text-slate-900 truncate">{title}</div>
              <div className="text-xs text-slate-600 mt-0.5">AskDB • Proof-first analytics</div>
            </div>
            <Button onClick={onClose} variant="soft" className="shrink-0">
              Close
            </Button>
          </div>
          <div className="mt-4">{children}</div>
        </Card>
      </div>
    </div>
  );
}

function HeroArt() {
  return (
    <div className="relative overflow-hidden rounded-3xl border border-slate-900/10 bg-white/70 backdrop-blur-xl shadow-[0_24px_70px_-45px_rgba(15,23,42,0.35)]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_15%_20%,rgba(56,189,248,0.35),transparent_50%),radial-gradient(circle_at_80%_10%,rgba(99,102,241,0.35),transparent_55%),radial-gradient(circle_at_55%_90%,rgba(217,70,239,0.25),transparent_55%)]" />
      <div className="absolute inset-0 opacity-[0.18] bg-[linear-gradient(to_right,rgba(15,23,42,0.45)_1px,transparent_1px),linear-gradient(to_bottom,rgba(15,23,42,0.45)_1px,transparent_1px)] bg-[size:44px_44px]" />
      <div className="absolute left-6 top-5 z-10">
        <div className="text-sm font-semibold text-slate-900 tracking-tight">AskDB in action</div>
        <div className="mt-0.5 text-xs text-slate-600">Question → SQL → Preview rows → Insight</div>
      </div>
      <svg viewBox="0 0 1200 420" className="relative block w-full h-[220px]" aria-hidden="true">
        <defs>
          <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="rgba(79,70,229,0.75)" />
            <stop offset="0.55" stopColor="rgba(217,70,239,0.55)" />
            <stop offset="1" stopColor="rgba(14,165,233,0.55)" />
          </linearGradient>
          <filter id="blur" x="-10%" y="-10%" width="120%" height="120%">
            <feGaussianBlur stdDeviation="14" />
          </filter>
        </defs>
        <path
          d="M80 300 C 260 80, 520 70, 690 220 C 820 335, 1040 350, 1140 170"
          fill="none"
          stroke="url(#g1)"
          strokeWidth="20"
          strokeLinecap="round"
          filter="url(#blur)"
          opacity="0.85"
        />
        <path
          d="M80 300 C 260 80, 520 70, 690 220 C 820 335, 1040 350, 1140 170"
          fill="none"
          stroke="rgba(15,23,42,0.18)"
          strokeWidth="4"
          strokeLinecap="round"
        />

        {[
          [140, 240],
          [300, 150],
          [520, 120],
          [720, 240],
          [920, 320],
          [1080, 220],
        ].map(([x, y], i) => (
          <g key={i}>
            <circle cx={x} cy={y} r="18" fill="rgba(255,255,255,0.95)" />
            <circle cx={x} cy={y} r="18" fill="none" stroke="rgba(15,23,42,0.15)" />
            <circle cx={x} cy={y} r="6" fill="rgba(79,70,229,0.65)" />
          </g>
        ))}
      </svg>

      <div className="relative px-6 pb-6">
        <div className="flex flex-wrap gap-2">
          <Pill tone="info">
            <Icon name="spark" className="h-4 w-4" />
            Proof-first output
          </Pill>
          <Pill>
            <Icon name="shield" className="h-4 w-4" />
            Guardrails + timeouts
          </Pill>
          <Pill>
            <Icon name="db" className="h-4 w-4" />
            BYODB optional
          </Pill>
        </div>
      </div>
    </div>
  );
}

// ---------------- main ----------------
export default function App() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("public");
  const [includeSql, setIncludeSql] = useState(true);
  const [loading, setLoading] = useState(false);
  const [job, setJob] = useState(null); // { id, status }

  const [about, setAbout] = useState(null);
  const [examples, setExamples] = useState([]);
  const [schema, setSchema] = useState([]);
  const [showSchema, setShowSchema] = useState(false);

  const [resp, setResp] = useState(null);
  const [error, setError] = useState("");

  const [phase, setPhase] = useState("idle"); // idle | sql | db | answer
  const [previewQuery, setPreviewQuery] = useState("");

  const [toast, setToast] = useState({ open: false, message: "" });
  const toastTimerRef = useRef(null);

  const [showConnect, setShowConnect] = useState(false);
  const [dbUrl, setDbUrl] = useState("");
  const [schemaCsv, setSchemaCsv] = useState("");
  const [connecting, setConnecting] = useState(false);

  const [demoKey, setDemoKey] = useState(() => localStorage.getItem("askdb_demo_key") || "");
  const [conn, setConn] = useState({
    connected: false,
    host: "",
    dialect: "",
    schema_source: "",
  });

  const [savedOpen, setSavedOpen] = useState(false);
  const [saved, setSaved] = useState(() => loadSavedQueries());

  // Tour
  const [showTourBanner, setShowTourBanner] = useState(false);
  const [tourStep, setTourStep] = useState(0); // 0 off, 1..3
  const [tourRect, setTourRect] = useState(null);
  const [tourCalloutPos, setTourCalloutPos] = useState({ top: 80, left: 24 });
  const [tourPendingKey, setTourPendingKey] = useState(false);
  const [resumeTourStep, setResumeTourStep] = useState(null);

  const resultsRef = useRef(null);
  const askBoxRef = useRef(null);
  const modeSelectRef = useRef(null);
  const evidenceRef = useRef(null);

  const sessionId = useMemo(() => {
    const key = "askdb_session_id";
    let id = localStorage.getItem(key);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(key, id);
    }
    return id;
  }, []);

  // persist demo key
  useEffect(() => {
    localStorage.setItem("askdb_demo_key", demoKey || "");
  }, [demoKey]);

  // first visit banner
  useEffect(() => {
    const seen = localStorage.getItem("askdb_tour_seen_v3");
    if (!seen) setShowTourBanner(true);
  }, []);

  // auto-run share link
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q");
    const m = params.get("mode");
    const run = params.get("run");
    if (q) setQuestion(q);
    if (m === "public" || m === "sandbox") setMode(m);
    if (q && run === "1") setTimeout(() => ask(false, q, m), 350);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshStatus() {
    try {
      const c = await fetch(`${API_BASE}/connection?session_id=${encodeURIComponent(sessionId)}`);
      const cj = await c.json();
      setConn(cj?.connected ? cj : { connected: false, host: "", dialect: "", schema_source: "" });

      const s = await fetch(`${API_BASE}/schema?session_id=${encodeURIComponent(sessionId)}`);
      const sj = await s.json();
      setSchema(sj.tables || []);
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    (async () => {
      try {
        const a = await fetch(`${API_BASE}/about`);
        setAbout(await a.json());

        const e = await fetch(`${API_BASE}/examples`);
        const ej = await e.json();
        setExamples(ej.examples || []);

        await refreshStatus();
      } catch {
        // ignore
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // auto-scroll to results
  useEffect(() => {
    if (resp && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [resp]);

  function showToast(message) {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    setToast({ open: true, message });
    toastTimerRef.current = setTimeout(() => setToast({ open: false, message: "" }), 1800);
  }

  // spotlight helpers
  function clamp(n, min, max) {
    return Math.max(min, Math.min(max, n));
  }

  function computeSpotlight(step) {
    const el = step === 1 ? askBoxRef.current : step === 2 ? modeSelectRef.current : (evidenceRef.current || resultsRef.current);
    if (!el) return;

    const r = el.getBoundingClientRect();
    const pad = 10;
    const rect = {
      top: Math.max(12, r.top - pad),
      left: Math.max(12, r.left - pad),
      width: Math.max(20, r.width + pad * 2),
      height: Math.max(20, r.height + pad * 2),
    };
    setTourRect(rect);

    const cardW = 420;
    const cardH = 240;
    const gap = 14;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let left = rect.left + rect.width + gap;
    let top = rect.top;

    if (left + cardW > vw - 12) left = rect.left - cardW - gap;
    left = clamp(left, 12, vw - cardW - 12);

    if (top + cardH > vh - 12) top = rect.top + rect.height - cardH;
    top = clamp(top, 12, vh - cardH - 12);

    setTourCalloutPos({ top, left });
  }

  useEffect(() => {
    if (tourStep <= 0) {
      setTourRect(null);
      return;
    }
    const t = setTimeout(() => computeSpotlight(tourStep), 50);
    const onResize = () => computeSpotlight(tourStep);
    window.addEventListener("resize", onResize);
    window.addEventListener("scroll", onResize, true);
    return () => {
      clearTimeout(t);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("scroll", onResize, true);
    };
  }, [tourStep, resp, includeSql, showConnect]);

  // resume tour after demo key
  useEffect(() => {
    if (!resumeTourStep) return;
    if (!demoKey?.trim()) return;
    setShowConnect(false);
    setTourPendingKey(false);
    setTourStep(resumeTourStep);
    setResumeTourStep(null);
    showToast("Demo key saved — resuming tour");
  }, [demoKey, resumeTourStep]);

  
  async function pollJob(jobId) {
    let attempts = 0;
    let delay = 900;

    setJob({ id: jobId, status: "queued" });

    while (attempts < 120) { // ~2 minutes
      attempts += 1;
      try {
        const jr = await fetch(`${API_BASE}/job/${jobId}`, {
          headers: { ...(demoKey ? { "X-DEMO-KEY": demoKey } : {}) },
        });
        const jd = await jr.json();

        if (!jr.ok) throw new Error(jd?.error || "Job failed");

        if (jd.status === "finished") {
          setJob({ id: jobId, status: "finished" });
          setResp(jd.result);
          return;
        }

        if (jd.status === "failed") {
          setJob({ id: jobId, status: "failed" });
          throw new Error(jd?.error || "Job failed");
        }

        setJob({ id: jobId, status: jd.status || "running" });
      } catch (e) {
        setJob({ id: jobId, status: "failed" });
        setError(e.message || String(e));
        return;
      }

      await new Promise((res) => setTimeout(res, delay));
      delay = Math.min(2500, Math.floor(delay * 1.12));
    }

    setError("This query is taking longer than expected. Please try again.");
    setJob(null);
  }

async function ask(reset_session = false, qOverride = null, modeOverride = null) {
    setError("");
    setPreviewQuery("");

    const qToSend = (qOverride ?? question).trim();
    if (!qToSend) return;

    const effMode = modeOverride || mode;

    setLoading(true);
    setPhase("sql");
    const t1 = setTimeout(() => setPhase("db"), 650);
    const t2 = setTimeout(() => setPhase("answer"), 1350);

    try {
      const r = await fetch(`${API_BASE}/api`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(demoKey ? { "X-DEMO-KEY": demoKey } : {}),
        },
        body: JSON.stringify({
          question: qToSend,
          session_id: sessionId,
          mode: effMode,
          include_sql: includeSql,
          reset_session,
        }),
      });

      const data = await r.json();

      // Async mode: backend returns 202 + job_id
      if (r.status === 202 && data?.job_id) {
        clearTimeout(t1);
        clearTimeout(t2);
        setPhase("answer");
        await pollJob(data.job_id);
        return;
      }

      if (!r.ok) throw new Error(data?.error || "Request failed");

      setResp(data);
      setMode(effMode);
      await refreshStatus();
    } catch (e) {
      setResp(null);
      setError(e.message || String(e));
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      setPhase("idle");
      setLoading(false);
    }
  }

  async function copyToClipboard(text, msg = "Copied") {
    if (!text) return;
    await navigator.clipboard.writeText(text);
    showToast(msg);
  }

  function exportCSV() {
    if (!resp?.rows_preview?.length) return;
    const blob = downloadCSVBlob(resp.rows_preview);
    if (!blob) return;

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "askdb_preview.csv";
    a.click();
    URL.revokeObjectURL(url);
    showToast("Exported CSV");
  }

  function saveQuery() {
    const q = question.trim();
    if (!q) return;
    const item = { id: crypto.randomUUID(), q, mode, ts: Date.now() };
    const next = [item, ...saved].slice(0, 40);
    setSaved(next);
    saveSavedQueries(next);
    showToast("Saved");
  }

  function deleteSaved(id) {
    const next = saved.filter((x) => x.id !== id);
    setSaved(next);
    saveSavedQueries(next);
  }

  async function connectDb() {
    setConnecting(true);
    setError("");

    try {
      // If DB URL empty => treat as Demo DB selection (no network call needed).
      if (!dbUrl.trim()) {
        setShowConnect(false);
        showToast("Using Demo DB");
        await refreshStatus();
        return;
      }

      const r = await fetch(`${API_BASE}/connect`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(demoKey ? { "X-DEMO-KEY": demoKey } : {}),
        },
        body: JSON.stringify({
          session_id: sessionId,
          db_url: dbUrl.trim(),
          schema_csv: schemaCsv.trim() || undefined,
        }),
      });

      const data = await r.json();
      if (!r.ok) throw new Error(data?.error || "Connect failed");

      setShowConnect(false);
      showToast("Connected");
      await refreshStatus();
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setConnecting(false);
    }
  }

  async function disconnectDb() {
    setConnecting(true);
    setError("");

    try {
      const r = await fetch(`${API_BASE}/disconnect`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(demoKey ? { "X-DEMO-KEY": demoKey } : {}),
        },
        body: JSON.stringify({ session_id: sessionId }),
      });

      const data = await r.json();
      if (!r.ok) throw new Error(data?.error || "Disconnect failed");

      setShowConnect(false);
      showToast("Disconnected — back to Demo DB");
      await refreshStatus();
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setConnecting(false);
    }
  }

  // guided demo
  async function startTour() {
    localStorage.setItem("askdb_tour_seen_v3", "1");
    setShowTourBanner(false);
    setTourPendingKey(false);

    const q1 = "Top 10 customers by total payments and their share of total revenue";
    setMode("public");
    setQuestion(q1);
    setTourStep(1);
    askBoxRef.current?.scrollIntoView?.({ behavior: "smooth", block: "center" });
    await ask(false, q1, "public");
    setTourStep(2);
  }

  async function tourNext() {
    if (tourStep === 2) {
      if (!demoKey?.trim()) {
        setTourPendingKey(true);
        setResumeTourStep(2);
        setTourStep(0); // pause tour so connect modal is on top
        setShowConnect(true);
        showToast("Enter Demo Key to unlock sandbox");
        return;
      }

      const q2 = "In sandbox mode, delete order 10100 safely (delete orderdetails first, then the order)";
      setMode("sandbox");
      setQuestion(q2);
      setTourStep(2);
      modeSelectRef.current?.scrollIntoView?.({ behavior: "smooth", block: "center" });
      await ask(false, q2, "sandbox");
      setTourStep(3);
      return;
    }

    if (tourStep === 3) {
      setTourStep(0);
      showToast("Tour completed");
      return;
    }
  }

  function tourBack() {
    if (tourStep === 3) setTourStep(2);
    if (tourStep === 2) setTourStep(1);
  }

  function skipTour() {
    localStorage.setItem("askdb_tour_seen_v3", "1");
    setShowTourBanner(false);
    setTourStep(0);
  }

  // answer formatting
  const rawAnswer = String(resp?.answer || "");
  const lines = rawAnswer.split("\n").map(normalizeLine).filter(Boolean);
  const summary = lines[0] || "";
  const insightLines = lines.slice(1, 10);

  const previewRows = resp?.rows_preview || [];
  const filteredRows = previewRows.filter((r) => lineSearchMatch(r, previewQuery));
  const columns = resp?.columns?.length ? resp.columns : previewRows[0] ? Object.keys(previewRows[0]) : [];
  const showingCount = filteredRows.length;
  const totalCount = resp?.rows_returned ?? previewRows.length;

  const quickPrompts = [
    { label: "Top customers + revenue share", q: "Top 10 customers by total payments and their share of total revenue", mode: "public" },
    { label: "Revenue by product line", q: "Total sales by product line (last 90 days)", mode: "public" },
    { label: "Customers w/ no payments", q: "List customers who have placed orders but have never made a payment, including their total order count", mode: "public" },
    { label: "Sandbox: delete order safely", q: "In sandbox mode, delete order 10100 safely (delete orderdetails first, then the order)", mode: "sandbox" },
  ];

  return (
    <div className="min-h-screen text-slate-900">
      {/* background */}
      <style>{`
        @keyframes floaty { 0%{ transform: translate3d(0,0,0); } 50%{ transform: translate3d(0,-16px,0); } 100%{ transform: translate3d(0,0,0);} }
        .noise { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='220' height='220'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='220' height='220' filter='url(%23n)' opacity='.14'/%3E%3C/svg%3E"); }
      `}</style>

      <div className="fixed inset-0 -z-10 bg-[#F7F7FB]" />
      <div className="fixed inset-0 -z-10 opacity-30 bg-[radial-gradient(circle_at_15%_10%,rgba(56,189,248,0.35),transparent_40%),radial-gradient(circle_at_80%_15%,rgba(99,102,241,0.30),transparent_45%),radial-gradient(circle_at_55%_95%,rgba(217,70,239,0.22),transparent_50%)]" />
      <div className="fixed inset-0 -z-10 opacity-[0.10] noise mix-blend-multiply" />
      <div className="fixed -z-10 left-[-140px] top-[-140px] h-[520px] w-[520px] rounded-full bg-indigo-400/25 blur-[90px] animate-[floaty_8s_ease-in-out_infinite]" />
      <div className="fixed -z-10 right-[-140px] top-[120px] h-[560px] w-[560px] rounded-full bg-sky-400/18 blur-[95px] animate-[floaty_10s_ease-in-out_infinite]" />

      {/* Toast */}
      <div
        className={`fixed top-5 left-1/2 -translate-x-1/2 z-[95] transition ${
          toast.open ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        <div className="rounded-full border border-slate-900/10 bg-white/80 backdrop-blur-xl px-4 py-2 text-sm text-slate-900 shadow-[0_24px_80px_-55px_rgba(15,23,42,0.35)]">
          <span className="inline-block w-2 h-2 rounded-full bg-sky-500 mr-2 align-middle" />
          {toast.message}
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* NAV */}
        <Card tone="soft" className="px-5 py-4">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <BrandMark />

            {/* status */}
            <div className="flex flex-wrap items-center gap-2">
              {!conn.connected ? (
                <Pill>
                  <Icon name="db" className="h-4 w-4" />
                  Demo DB
                </Pill>
              ) : (
                <Pill tone="good">
                  <Icon name="db" className="h-4 w-4" />
                  Connected • {conn.host || "db"} • schema:{conn.schema_source || "auto"}
                </Pill>
              )}
              <Pill>
                <Icon name="shield" className="h-4 w-4" />
                public = SELECT-only
              </Pill>
              <Pill tone="warn">
                <Icon name="wand" className="h-4 w-4" />
                sandbox = rollback
              </Pill>
            </div>

            {/* buttons layout: 2 + 2 + centered 1 */}
            <div className="grid gap-2">
              <div className="grid grid-cols-2 gap-2">
                <Button variant="soft" onClick={() => setShowSchema(true)} className="w-full">
                  <Icon name="grid" className="h-4 w-4" />
                  Schema
                </Button>
                <Button variant="soft" onClick={() => setShowConnect(true)} className="w-full">
                  <Icon name="db" className="h-4 w-4" />
                  Connect DB
                </Button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Button variant="soft" onClick={disconnectDb} disabled={!conn.connected || connecting} className="w-full">
                  Disconnect
                </Button>
                <Button variant="soft" onClick={() => ask(true)} className="w-full">
                  Reset session
                </Button>
              </div>
              <div className="flex justify-center">
                <Button variant="soft" onClick={() => setSavedOpen(true)} className="w-full max-w-[260px]">
                  <Icon name="link" className="h-4 w-4" />
                  Saved / Share
                </Button>
              </div>
            </div>
          </div>
        </Card>

        {/* HERO GRID */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7">
            <HeroArt />
          </div>

  <div className="lg:col-span-5 grid gap-6">
    {/* Primary action (above the fold) */}
    <Card tone="soft" className="p-6">
      <div className="flex flex-col gap-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[11px] uppercase tracking-widest text-slate-500">Command</div>
            <div className="mt-1 text-lg font-semibold tracking-tight">Ask your database</div>
            <div className="text-sm text-slate-600 mt-1">
              Ask in plain English. We generate SQL, run it safely, and show the proof.
            </div>
          </div>
          <Pill tone={mode === "sandbox" ? "warn" : "default"}>
            <Icon name={mode === "sandbox" ? "wand" : "shield"} className="h-4 w-4" />
            {mode === "sandbox" ? "sandbox • rollback" : "public • SELECT-only"}
          </Pill>
        </div>

        <div className="relative">
          <div className="absolute inset-0 rounded-3xl bg-[radial-gradient(circle_at_30%_20%,rgba(99,102,241,0.22),transparent_55%),radial-gradient(circle_at_80%_40%,rgba(56,189,248,0.18),transparent_55%)] blur-[18px]" />
          <textarea
            ref={askBoxRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            placeholder="e.g., Top 10 customers by total payments and revenue share (last 90 days)"
            className="relative w-full rounded-3xl border border-slate-900/10 bg-white px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/15"
          />
        </div>

        {/* Quick prompts */}
        <div className="flex flex-wrap gap-2">
          {quickPrompts.map((p) => (
            <button
              key={p.label}
              onClick={() => {
                setMode(p.mode);
                setQuestion(p.q);
                ask(false, p.q, p.mode);
              }}
              className="inline-flex items-center gap-2 rounded-full border border-slate-900/10 bg-white hover:bg-slate-50 active:bg-slate-100 px-3 py-1.5 text-[12px] font-semibold text-slate-700 transition"
              title={p.q}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-indigo-600/70" />
              {p.label}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-1">
          <select
            ref={modeSelectRef}
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            className="h-10 rounded-2xl border border-slate-900/10 bg-white px-3 text-sm font-semibold text-slate-900"
          >
            <option value="public">public (SELECT-only)</option>
            <option value="sandbox">sandbox (DML simulated)</option>
          </select>

          <label className="flex items-center gap-2 text-sm text-slate-700">
            <input type="checkbox" checked={includeSql} onChange={(e) => setIncludeSql(e.target.checked)} />
            Show SQL + preview
          </label>

          <Button variant="soft" onClick={saveQuery}>
            Save
          </Button>
          <Button variant="primary" onClick={() => ask(false)} disabled={loading} className="ml-auto">
            {loading ? "Running…" : "Run"}
          </Button>
        </div>

        {/* Stepper + shimmer */}
        {loading && (
          <div className="mt-1">
            <div className="h-2 rounded-full bg-slate-900/10 overflow-hidden">
              <div
                className={`h-full bg-indigo-600 transition-all duration-500 ${
                  phase === "sql" ? "w-1/4" : phase === "db" ? "w-2/4" : phase === "answer" ? "w-3/4" : "w-0"
                }`}
              />
            </div>
            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-600">
              <Pill>Generating SQL</Pill>
              <Pill>Querying DB</Pill>
              <Pill>Writing answer</Pill>
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 p-3 text-rose-900">
            <b>Error:</b> {error}
          </div>
        )}
      </div>
    </Card>

    <Card tone="soft" className="p-6">
      <div className="flex items-center gap-2 text-slate-900">
        <Icon name="shield" className="h-5 w-5" />
        <div className="font-semibold">Safety Model</div>
      </div>
      <div className="mt-3 text-sm text-slate-600 leading-relaxed">
        <b className="text-slate-900">Public</b>: analytics only (SELECT) with limits + timeouts.
        <br />
        <b className="text-slate-900">Sandbox</b>: lets users try INSERT/UPDATE/DELETE, but <b className="text-slate-900">always rolls back</b>.
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <Pill>0 permanent writes</Pill>
        <Pill>DDL blocked</Pill>
      </div>
    </Card>

    <Card tone="soft" className="p-6">
      <div className="flex items-center gap-2 text-slate-900">
        <Icon name="db" className="h-5 w-5" />
        <div className="font-semibold">Bring Your Own DB</div>
      </div>
      <div className="mt-3 text-sm text-slate-600 leading-relaxed">
        Connect any Postgres/MySQL/SQLite URL. Schema CSV is optional — AskDB can auto-detect schema.
      </div>
      <div className="mt-4 flex gap-2">
        <Button variant="primary" onClick={() => setShowConnect(true)}>
          Connect
        </Button>
        {showTourBanner ? (
          <>
            <Button variant="soft" onClick={startTour}>
              <Icon name="spark" className="h-4 w-4" />
              Guided demo
            </Button>
            <Button variant="soft" onClick={skipTour}>
              Skip
            </Button>
          </>
        ) : (
          <Button
            variant="soft"
            onClick={() => {
              localStorage.removeItem("askdb_tour_seen_v3");
              setShowTourBanner(true);
              showToast("Demo banner re-enabled");
            }}
          >
            Re-enable demo banner
          </Button>
        )}
      </div>
      <div className="mt-3 text-xs text-slate-500">Session: {sessionId}</div>
    </Card>
  </div>
</div>


        {/* RESULTS */}
        {resp && (
          <div ref={resultsRef} className="mt-6 grid gap-6">
            <Card tone="soft" className="p-6">
              <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
                <div>
                  <div className="text-[11px] uppercase tracking-widest text-slate-500">Results</div>
                  <div className="mt-1 text-lg font-semibold">Answer + Proof</div>
                  <div className="text-sm text-slate-600 mt-1">Transparent SQL, preview rows, export/share.</div>
                </div>
                {resp.mode === "sandbox" && resp.rolled_back && (
                  <Pill tone="warn">
                    <Icon name="wand" className="h-4 w-4" />
                    Sandbox simulation — rolled back
                  </Pill>
                )}
              </div>

              {/* Answer */}
              <div className="mt-5 grid gap-5">
                <Card tone="plain" className="p-5">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <div className="text-sm font-semibold text-slate-900">Answer</div>
                    <div className="text-xs text-slate-500">
                      {resp.db_ms != null ? (
                        <>
                          db: <b className="text-slate-900">{resp.db_ms}ms</b> •{" "}
                        </>
                      ) : null}
                      total: <b className="text-slate-900">{resp.latency_ms}ms</b>
                      {resp.optimized ? (
                        <span className="ml-2 px-2 py-0.5 rounded-full bg-indigo-600/10 border border-indigo-600/15 text-indigo-900 text-[11px]">
                          optimized
                        </span>
                      ) : null}
                    </div>
                  </div>

                  <div className="mt-3 text-slate-900 leading-relaxed text-[15px]">
                    {summary || <span className="text-slate-500">(No summary returned — try another query.)</span>}
                  </div>

                  {resp.insights_summary && (
                    <div className="mt-4 rounded-2xl border border-slate-900/10 bg-slate-50 p-4">
                      <div className="text-[11px] uppercase tracking-widest text-slate-500">AI insight</div>
                      <div className="mt-2 text-sm text-slate-700 leading-relaxed">{resp.insights_summary}</div>
                    </div>
                  )}

                  {insightLines.length > 0 && (
                    <div className="mt-4 grid gap-2">
                      {insightLines.map((line, i) => (
                        <div
                          key={i}
                          className="flex items-start gap-3 rounded-2xl border border-slate-900/10 bg-slate-50 px-4 py-3"
                        >
                          <span className="mt-1.5 h-2 w-2 rounded-full bg-sky-500 shrink-0" />
                          <span className="text-slate-700 text-sm leading-relaxed">{line}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  <details className="mt-4">
                    <summary className="cursor-pointer text-sm text-slate-600 hover:text-slate-900 transition">
                      View full answer (raw)
                    </summary>
                    <div className="mt-3 prose max-w-none prose-slate prose-p:text-slate-700 prose-strong:text-slate-900">
                      <ReactMarkdown>{resp.answer}</ReactMarkdown>
                    </div>
                  </details>
                </Card>

                {/* Chart */}
                {resp.chart_spec && resp.rows_preview?.length ? <Chart spec={resp.chart_spec} rows={resp.rows_preview} /> : null}

                {/* Evidence */}
                {includeSql && (
                  <Card tone="plain" className="p-5" >
                    <div ref={evidenceRef}>
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="text-sm font-semibold text-slate-900">SQL & Preview</div>
                        <div className="flex flex-wrap gap-2">
                          <Button variant="soft" onClick={() => copyToClipboard(resp.sql, "Copied SQL")} disabled={!resp.sql}>
                            Copy SQL
                          </Button>
                          <Button variant="soft" onClick={exportCSV} disabled={!resp.rows_preview?.length}>
                            Export CSV
                          </Button>
                          <Button
                            variant="soft"
                            onClick={() => copyToClipboard(encodeShareUrl({ q: question.trim(), mode }), "Copied share link")}
                          >
                            Share
                          </Button>
                        </div>
                      </div>

                      <div className="mt-3 grid grid-cols-1 md:grid-cols-4 gap-3 text-[11px]">
                        {[
                          ["mode", resp.mode],
                          ["kind", resp.kind],
                          ["rows", String(resp.rows_returned ?? previewRows.length)],
                          ["latency", `${resp.latency_ms} ms`],
                        ].map(([k, v]) => (
                          <div key={k} className="rounded-2xl border border-slate-900/10 bg-slate-50 px-4 py-3 text-slate-700">
                            <div className="uppercase tracking-widest text-slate-500">{k}</div>
                            <div className="mt-1 text-slate-900 font-semibold">{v}</div>
                          </div>
                        ))}
                      </div>

                      <pre className="mt-4 overflow-auto rounded-3xl bg-slate-950 text-slate-100 p-5 text-xs border border-slate-900/10 whitespace-pre-wrap">
{prettySql(resp.sql || "")}
                      </pre>

                      <div className="mt-4 flex flex-col md:flex-row md:items-end md:justify-between gap-3">
                        <div>
                          <div className="text-sm font-semibold text-slate-900">Preview</div>
                          <div className="text-xs text-slate-600 mt-1">
                            Showing <span className="text-slate-900 font-semibold">{showingCount}</span> of{" "}
                            <span className="text-slate-900 font-semibold">{totalCount}</span> rows
                          </div>
                        </div>

                        <div className="w-full md:w-80">
                          <input
                            value={previewQuery}
                            onChange={(e) => setPreviewQuery(e.target.value)}
                            placeholder="Search preview…"
                            className="w-full rounded-2xl border border-slate-900/10 bg-white px-4 py-3 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/15"
                          />
                        </div>
                      </div>

                      {filteredRows.length ? (
                        <div className="mt-3 overflow-auto rounded-3xl border border-slate-900/10 bg-white">
                          <table className="w-full text-xs">
                            <thead className="bg-slate-50 sticky top-0">
                              <tr>
                                {columns.map((c) => (
                                  <th key={c} className="text-left p-4 border-b border-slate-900/10 text-slate-700">
                                    {c}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {filteredRows.map((row, idx) => (
                                <tr key={idx} className="odd:bg-white even:bg-slate-50/60">
                                  {columns.map((c) => (
                                    <td key={c} className="p-4 border-b border-slate-900/10 text-slate-700">
                                      {row[c] === null || row[c] === undefined ? "" : String(row[c])}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <div className="mt-3 text-xs text-slate-600">No matching rows.</div>
                      )}
                    </div>
                  </Card>
                )}
              </div>
            </Card>
          </div>
        )}

        {/* EXAMPLES */}
        <Card tone="soft" className="mt-6 p-6">
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
            <div>
              <div className="text-[11px] uppercase tracking-widest text-slate-500">Try examples</div>
              <div className="mt-1 text-lg font-semibold">Curated prompts</div>
              <div className="text-sm text-slate-600 mt-1">Click a prompt to run instantly.</div>
            </div>
            <div className="text-xs text-slate-600">Sandbox may require Demo Key if enabled.</div>
          </div>

          <div className="mt-5 grid gap-5">
            {examples.map((group) => (
              <div key={group.category}>
                <div className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-indigo-600/70" />
                  {group.category}
                  {group.mode ? (
                    <span className="ml-2 text-[11px] px-2 py-0.5 rounded-full bg-slate-900/5 border border-slate-900/10 text-slate-700">
                      {group.mode}
                    </span>
                  ) : null}
                </div>

                <div className="mt-3 flex flex-wrap gap-2">
                  {(group.items || []).map((q) => (
                    <button
                      key={q}
                      onClick={() => {
                        const m = group.mode || "public";
                        setMode(m);
                        setQuestion(q);
                        ask(false, q, m);
                      }}
                      className="px-4 py-2 rounded-2xl border border-slate-900/10 bg-white hover:bg-slate-50 active:bg-slate-100 transition text-sm font-semibold text-slate-900"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            {!examples.length && <div className="text-slate-600 text-sm">Loading examples…</div>}
          </div>
        </Card>

        <div className="mt-10 text-center text-xs text-slate-500">
          Built like a product • Proof-first • BYODB-ready
        </div>
      </div>

      {/* MODALS */}
      {showSchema && (
        <ModalShell title="Schema" onClose={() => setShowSchema(false)}>
          <div className="max-h-[60vh] overflow-auto rounded-3xl border border-slate-900/10 bg-white">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 sticky top-0">
                <tr>
                  <th className="text-left p-4 border-b border-slate-900/10 text-slate-800">Table</th>
                  <th className="text-left p-4 border-b border-slate-900/10 text-slate-800">Description</th>
                </tr>
              </thead>
              <tbody>
                {schema.map((t) => (
                  <tr key={t.table_name} className="odd:bg-white even:bg-slate-50/60">
                    <td className="p-4 border-b border-slate-900/10 font-semibold text-slate-900">{t.table_name}</td>
                    <td className="p-4 border-b border-slate-900/10 text-slate-700">{t.description}</td>
                  </tr>
                ))}
                {!schema.length && (
                  <tr>
                    <td className="p-4 text-slate-600" colSpan={2}>
                      No schema loaded.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-slate-600 mt-3">If BYODB is connected, schema reflects your database.</p>
        </ModalShell>
      )}

      {showConnect && (
        <ModalShell title="Connect your database (BYODB)" onClose={() => setShowConnect(false)}>
          <div className="grid gap-3">
            <div>
              <div className="text-xs text-slate-600 mb-1 font-semibold">Demo Key (optional)</div>
              <input
                className="w-full rounded-2xl border border-slate-900/10 bg-white px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/15"
                value={demoKey}
                onChange={(e) => setDemoKey(e.target.value)}
                placeholder="X-DEMO-KEY (if required)"
              />
              <div className="text-xs text-slate-500 mt-1">If backend is gated, sandbox/BYODB requires this key.</div>
            </div>

            <div>
              <div className="text-xs text-slate-600 mb-1 font-semibold">Database URL</div>
              <input
                className="w-full rounded-2xl border border-slate-900/10 bg-white px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/15"
                value={dbUrl}
                onChange={(e) => setDbUrl(e.target.value)}
                placeholder="postgresql+psycopg2://user:pass@host:5432/db?sslmode=require"
              />
              <div className="text-xs text-slate-500 mt-1">Leave blank to use the Demo DB.</div>
            </div>

            <div>
              <div className="text-xs text-slate-600 mb-1 font-semibold">Schema CSV (optional)</div>
              <textarea
                className="w-full rounded-2xl border border-slate-900/10 bg-white px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/15"
                rows={6}
                value={schemaCsv}
                onChange={(e) => setSchemaCsv(e.target.value)}
                placeholder={`table_name,description\ncustomers,Contains customer info...`}
              />
              <div className="text-xs text-slate-500 mt-1">If omitted/invalid, AskDB auto-detects schema.</div>
            </div>

            <div className="flex flex-wrap gap-2 justify-end">
              <Button
                variant="soft"
                onClick={() => {
                  setDbUrl("");
                  setSchemaCsv("");
                  setShowConnect(false);
                  showToast("Using Demo DB");
                  refreshStatus();
                }}
                disabled={connecting}
              >
                Use Demo DB
              </Button>

              <Button variant="danger" onClick={disconnectDb} disabled={!conn.connected || connecting}>
                Disconnect
              </Button>

              <Button variant="primary" onClick={connectDb} disabled={connecting}>
                {connecting ? "Connecting…" : "Connect"}
              </Button>
            </div>

            {error && (
              <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 p-3 text-rose-900">
                <b>Error:</b> {error}
              </div>
            )}
          </div>
        </ModalShell>
      )}

      {savedOpen && (
        <ModalShell title="Saved queries" onClose={() => setSavedOpen(false)}>
          <div className="grid gap-2">
            {!saved.length && <div className="text-slate-600 text-sm">No saved queries yet.</div>}

            {saved.map((it) => (
              <div key={it.id} className="rounded-3xl border border-slate-900/10 bg-white p-4">
                <div className="text-sm text-slate-900 font-semibold">{it.q}</div>
                <div className="text-xs text-slate-500 mt-1">
                  {it.mode} • {new Date(it.ts).toLocaleString()}
                </div>

                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    variant="primary"
                    onClick={() => {
                      setMode(it.mode);
                      setQuestion(it.q);
                      setSavedOpen(false);
                      ask(false, it.q, it.mode);
                    }}
                  >
                    Run
                  </Button>
                  <Button
                    variant="soft"
                    onClick={() => copyToClipboard(encodeShareUrl({ q: it.q, mode: it.mode }), "Copied share link")}
                  >
                    Share
                  </Button>
                  <Button variant="danger" onClick={() => deleteSaved(it.id)}>
                    Delete
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </ModalShell>
      )}

      {/* Spotlight Tour */}
      {tourStep > 0 && tourRect && !showConnect && (
        <>
          <div
            className="fixed z-[75] pointer-events-none rounded-3xl border border-indigo-600/20"
            style={{
              top: tourRect.top,
              left: tourRect.left,
              width: tourRect.width,
              height: tourRect.height,
              boxShadow: "0 0 0 9999px rgba(15,23,42,0.30)",
            }}
          >
            <div className="absolute -inset-1 rounded-3xl ring-2 ring-indigo-500/25" />
          </div>

          <div className="fixed z-[76] pointer-events-auto" style={{ top: tourCalloutPos.top, left: tourCalloutPos.left, width: 420 }}>
            <Card tone="plain" className="p-5">
              <div className="flex items-center justify-between">
                <div className="text-[11px] uppercase tracking-widest text-slate-500">Guided demo</div>
                <div className="text-xs text-slate-500">{tourStep}/3</div>
              </div>

              <div className="mt-2 text-base font-semibold text-slate-900">
                {tourStep === 1 ? "Step 1 — Ask a question" : tourStep === 2 ? "Step 2 — Safe what-if (sandbox)" : "Step 3 — Proof & export"}
              </div>

              <div className="mt-2 text-sm text-slate-600 leading-relaxed">
                {tourStep === 1 && <>We ran an analytics query. Now look at the answer and verify it with SQL + preview rows.</>}
                {tourStep === 2 && <>Now we run a sandbox simulation (always rolled back). If required, enter Demo Key via Connect DB.</>}
                {tourStep === 3 && <>Copy SQL, export CSV, and share a reproducible link to the same query.</>}
              </div>

              <div className="mt-4 flex items-center justify-between gap-2">
                <Button
                  variant="soft"
                  onClick={() => {
                    setTourStep(0);
                    setTourPendingKey(false);
                    showToast("Tour closed");
                  }}
                >
                  Close
                </Button>

                <div className="flex gap-2">
                  {tourStep > 1 && (
                    <Button variant="soft" onClick={tourBack}>
                      Back
                    </Button>
                  )}
                  <Button variant="primary" onClick={tourNext}>
                    {tourStep === 3 ? "Finish" : "Next"}
                  </Button>
                </div>
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
