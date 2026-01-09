import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

const API_BASE = import.meta.env.VITE_API_BASE_URL;

// ---------- helpers ----------
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
    "SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY",
    "LIMIT", "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN",
    "OUTER JOIN", "UNION", "VALUES", "INSERT", "UPDATE", "DELETE",
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

function maskDbUrl(u) {
  const s = String(u || "");
  if (!s) return "";
  // hide password between ":" and "@"
  // e.g. postgresql://user:pass@host -> postgresql://user:••••@host
  return s.replace(/:\/\/([^:]+):([^@]+)@/, (m, user) => `://${user}:••••@`);
}

// ---------- main ----------
export default function App() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("public");
  const [includeSql, setIncludeSql] = useState(true);
  const [loading, setLoading] = useState(false);
  const [showInfo, setShowInfo] = useState(false);

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

  const resultsRef = useRef(null);

  // ---------------------------
  // V2: BYODB UI state
  // ---------------------------
  const [showConnect, setShowConnect] = useState(false);
  const [dbUrl, setDbUrl] = useState("");
  const [schemaCsv, setSchemaCsv] = useState(""); // text input / uploaded
  const [connecting, setConnecting] = useState(false);
  const [conn, setConn] = useState({ connected: false, ttl_seconds: 0 });

  const sessionId = useMemo(() => {
    const key = "askdb_session_id";
    let id = localStorage.getItem(key);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(key, id);
    }
    return id;
  }, []);

  // Fetch discovery endpoints
  useEffect(() => {
    (async () => {
      try {
        const a = await fetch(`${API_BASE}/about`);
        setAbout(await a.json());

        const e = await fetch(`${API_BASE}/examples`);
        const ej = await e.json();
        setExamples(ej.examples || []);

        const s = await fetch(`${API_BASE}/schema`);
        const sj = await s.json();
        setSchema(sj.tables || []);
      } catch {
        // ignore
      }
    })();
  }, []);

  // BYODB: fetch connection status
  async function refreshConnectionStatus() {
    try {
      const r = await fetch(`${API_BASE}/connection?session_id=${encodeURIComponent(sessionId)}`);
      const data = await r.json();
      if (!r.ok) return;
      setConn({
        connected: !!data.connected,
        ttl_seconds: data.ttl_seconds || 0,
        expires_at: data.expires_at,
      });
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    refreshConnectionStatus();
    // periodic refresh
    const t = setInterval(refreshConnectionStatus, 15000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Auto-scroll to results after response
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

  async function ask(reset_session = false, qOverride = null) {
    setError("");
    setPreviewQuery("");
    const qToSend = (qOverride ?? question).trim();
    if (!qToSend) return;

    setLoading(true);

    setPhase("sql");
    const t1 = setTimeout(() => setPhase("db"), 650);
    const t2 = setTimeout(() => setPhase("answer"), 1350);

    try {
      const r = await fetch(`${API_BASE}/api`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: qToSend,
          session_id: sessionId,
          mode,
          include_sql: includeSql,
          reset_session,
        }),
      });

      const data = await r.json();
      if (!r.ok) throw new Error(data?.error || "Request failed");

      setResp(data);

      // If backend reports BYODB status, update UI
      if (typeof data.byodb_connected === "boolean") {
        setConn((prev) => ({
          ...prev,
          connected: data.byodb_connected,
          ttl_seconds: data.byodb_ttl_seconds || prev.ttl_seconds,
        }));
      }
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

  async function copyToClipboard(text) {
    if (!text) return;
    await navigator.clipboard.writeText(text);
    showToast("Copied SQL");
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

  // ---------- Executive answer formatting ----------
  const rawAnswer = String(resp?.answer || "");
  const lines = rawAnswer.split("\n").map(normalizeLine).filter(Boolean);

  const summary = lines[0] || "";
  const insightLines = lines.slice(1, 10);

  const firstRow = resp?.rows_preview?.[0] || null;
  const chipKeys = firstRow ? Object.keys(firstRow).slice(0, 6) : [];
  const keyChips = firstRow ? chipKeys.map((k) => ({ k, v: firstRow[k] })) : [];

  const previewRows = resp?.rows_preview || [];
  const filteredRows = previewRows.filter((r) => lineSearchMatch(r, previewQuery));
  const columns = resp?.columns?.length ? resp.columns : (previewRows[0] ? Object.keys(previewRows[0]) : []);
  const showingCount = filteredRows.length;
  const totalCount = resp?.rows_returned ?? previewRows.length;

  // ---------- Apple-like design tokens ----------
  const card =
    "rounded-3xl border border-white/10 bg-white/[0.04] backdrop-blur-xl shadow-[0_18px_60px_-28px_rgba(0,0,0,0.85)]";
  const cardInner =
    "rounded-2xl border border-white/10 bg-black/20";

  const softBtn =
  "h-10 px-4 rounded-full border border-white/12 bg-white/[0.06] hover:bg-white/[0.10] active:bg-white/[0.12] transition text-white/85 whitespace-nowrap inline-flex items-center justify-center text-sm leading-none";
const dangerBtn =
  "h-10 px-4 rounded-full border border-white/12 bg-white/[0.06] hover:bg-red-500/15 active:bg-red-500/20 transition text-white/85 whitespace-nowrap inline-flex items-center justify-center text-sm leading-none";

  const primaryBtn =
    "px-5 py-2 rounded-full bg-[#0A84FF] text-white font-semibold hover:brightness-110 active:brightness-95 disabled:opacity-60 transition shadow-[0_10px_30px_-14px_rgba(10,132,255,0.75)]";

  const pill =
    "inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-xs text-white/75";

  const inputBase =
    "w-full rounded-2xl border border-white/10 bg-black/35 p-3 text-white placeholder:text-white/35 focus:outline-none focus:ring-4 focus:ring-[#0A84FF]/15 focus:border-[#0A84FF]/40";

  const selectBase =
    "rounded-2xl border border-white/10 px-3 py-2 bg-black/35 text-white focus:outline-none focus:ring-4 focus:ring-[#0A84FF]/15 focus:border-[#0A84FF]/40";


  // BYODB: file upload handler
  async function onUploadCsvFile(file) {
    if (!file) return;
    const text = await file.text();
    setSchemaCsv(text);
    showToast("Loaded CSV");
  }

  async function connectDb() {
    setError("");
    setConnecting(true);
    try {
      const payload = {
        session_id: sessionId,
        db_url: dbUrl.trim(),
        schema_csv: schemaCsv.trim(), // can be empty; backend will auto-generate schema if invalid/missing
      };

      const r = await fetch(`${API_BASE}/connect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await r.json();
      if (!r.ok) throw new Error(data?.error || "Connect failed");

      showToast("Connected");
      setShowConnect(false);
      await refreshConnectionStatus();

      // Optional: reset conversation when switching DB (avoid cross-DB context confusion)
      // (Your backend disconnect resets by default; connect doesn't. We can keep it simple.)
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setConnecting(false);
    }
  }

  async function disconnectDb() {
    setError("");
    setConnecting(true);
    try {
      const r = await fetch(`${API_BASE}/disconnect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, reset_session: true }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.error || "Disconnect failed");

      showToast("Disconnected");
      await refreshConnectionStatus();

      // Clear last response to avoid confusion
      setResp(null);
      setQuestion("");
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setConnecting(false);
    }
  }

  return (
    <div className="min-h-screen text-white">
      {/* Apple-like dark backdrop (neutral, minimal color) */}
      <div className="fixed inset-0 -z-10 bg-[#0B0B0F]" />
      <div className="fixed inset-0 -z-10 opacity-70 bg-[radial-gradient(circle_at_20%_15%,rgba(255,255,255,0.10),transparent_40%),radial-gradient(circle_at_70%_30%,rgba(10,132,255,0.12),transparent_45%)]" />
      <div className="fixed inset-0 -z-10 opacity-30 bg-[linear-gradient(to_bottom,rgba(255,255,255,0.05),transparent_30%,rgba(255,255,255,0.02))]" />

      {/* Toast */}
      <div
        className={`fixed top-4 left-1/2 -translate-x-1/2 z-50 transition ${
          toast.open ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        <div className="rounded-full border border-white/12 bg-black/55 backdrop-blur-xl px-4 py-2 text-sm text-white/90 shadow-[0_20px_70px_-35px_rgba(0,0,0,0.9)]">
          <span className="inline-block w-2 h-2 rounded-full bg-[#0A84FF] mr-2 align-middle" />
          {toast.message}
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 py-10">
        {/* Hero / Header */}
        <div className={`${card} p-7`}>
          <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6">
            <div className="min-w-0">
              <div className="flex flex-wrap gap-2">
                <span className={pill}>AskDB</span>
                <span className={pill}>Natural Language → SQL</span>
                <span className={pill}>Public + Sandbox</span>

                {/* BYODB status pill */}
                {conn.connected ? (
                  <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-[#0A84FF]/10 px-3 py-1 text-xs text-white/85">
                    <span className="h-2 w-2 rounded-full bg-[#0A84FF]" />
                    Connected DB
                    {conn.ttl_seconds ? (
                      <span className="text-white/60">({Math.max(0, Math.floor(conn.ttl_seconds / 60))}m)</span>
                    ) : null}
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-xs text-white/60">
                    <span className="h-2 w-2 rounded-full bg-white/30" />
                    Demo DB
                  </span>
                )}
              </div>

              <h1 className="mt-4 text-3xl md:text-4xl font-semibold tracking-tight leading-tight">
                Natural language → SQL → Business insight
                <span className="block text-white/60 text-xl md:text-2xl font-medium mt-2">
                  Safe “what-if” sandbox • Real database results
                </span>
              </h1>

              <p className="mt-4 text-white/60 leading-relaxed">
                Public mode is read-only analytics (SELECT-only). Sandbox mode simulates updates/deletes and always rolls back.
              </p>

              <div className="mt-5 text-sm text-white/60 space-y-1">
                {about ? (
                  <>
                    <div className="truncate">
                      <span className="text-white/85 font-semibold">Dataset:</span>{" "}
                      {about.dataset?.name}
                    </div>
                    <div className="truncate">
                      <span className="text-white/85 font-semibold">Entities:</span>{" "}
                      {(about.dataset?.entities || []).join(", ")}
                    </div>
                  </>
                ) : (
                  <div className="text-white/45">Connecting to backend…</div>
                )}
              </div>

              <p className="mt-4 text-xs text-white/35">Session: {sessionId}</p>
            </div>

            <div className="grid gap-2">
  {/* Row 1 */}
  <div className="grid grid-cols-2 gap-2 ">
    <button onClick={() => setShowSchema(true)} className={softBtn}>
      View schema
    </button>
    <button onClick={() => setShowConnect(true)} className={softBtn}>
      Connect DB
    </button>
  </div>

  {/* Row 2 */}
  <div className="grid grid-cols-2 gap-2 justify-self-end">
    <button
      onClick={disconnectDb}
      className={dangerBtn}
      disabled={!conn.connected || connecting}
      title={!conn.connected ? "No BYODB connection to disconnect" : "Disconnect and go back to demo DB"}
    >
      {connecting ? "…" : "Disconnect"}
    </button>

    <button onClick={() => ask(true)} className={softBtn} title="Clear conversation context">
      Reset session
    </button>
  </div>

  {/* Row 3 (centered) */}
  <div className="flex justify-center">
    <button onClick={() => setShowInfo(true)} className={softBtn}>
      Project info
    </button>
  </div>
</div>


          </div>
        </div>

        {/* Ask */}
        <div className={`mt-6 ${card} p-7`}>
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold tracking-tight">Ask a question</h2>
              <p className="text-sm text-white/55 mt-1">
                Start with an analytics question. Switch to sandbox to safely simulate writes.
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 rounded-full border border-white/10 bg-white/[0.05] text-xs text-white/70">
                public: SELECT-only
              </span>
              <span className="px-3 py-1 rounded-full border border-white/10 bg-white/[0.05] text-xs text-white/70">
                sandbox: rollback
              </span>
            </div>
          </div>

          <div className="mt-5 grid gap-4">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={3}
              placeholder="e.g., Top 3 customers by total payments"
              className={inputBase}
            />

            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2">
                <span className="text-sm text-white/70">Mode</span>
                <select value={mode} onChange={(e) => setMode(e.target.value)} className={selectBase}>
                  <option value="public">public (SELECT-only)</option>
                  <option value="sandbox">sandbox (DML simulated)</option>
                </select>
              </label>

              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={includeSql}
                  onChange={(e) => setIncludeSql(e.target.checked)}
                />
                <span className="text-sm text-white/70">Show SQL + preview</span>
              </label>

              <button onClick={() => ask(false)} disabled={loading} className={`ml-auto ${primaryBtn}`}>
                {loading ? "Asking…" : "Ask"}
              </button>
            </div>

            {loading && (
              <div className="mt-1">
                <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className={`h-full bg-[#0A84FF]/90 transition-all duration-500 ${
                      phase === "sql" ? "w-1/4" : phase === "db" ? "w-2/4" : phase === "answer" ? "w-3/4" : "w-0"
                    }`}
                  />
                </div>

                <div className="mt-3 flex flex-wrap gap-2 text-xs">
                  {[
                    ["sql", "Generating SQL"],
                    ["db", "Querying DB"],
                    ["answer", "Writing answer"],
                  ].map(([k, label]) => (
                    <span
                      key={k}
                      className={`px-3 py-1 rounded-full border transition ${
                        phase === k
                          ? "border-[#0A84FF]/35 bg-[#0A84FF]/10 text-white/90"
                          : "border-white/10 bg-white/[0.03] text-white/45"
                      }`}
                    >
                      {label}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {error && (
              <div className="rounded-2xl border border-red-400/20 bg-red-500/10 p-3 text-red-200">
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Results */}
        {resp && (
          <div ref={resultsRef} className={`mt-6 ${card} p-7`}>
            <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold tracking-tight">Results</h2>
                <p className="text-sm text-white/55 mt-1">
                  Executive summary • SQL transparency • Preview table
                </p>
              </div>

              {resp.mode === "sandbox" && resp.rolled_back && (
                <div className="rounded-full border border-white/10 bg-white/[0.05] px-4 py-2 text-xs text-white/80">
                  ✅ Sandbox simulation — rolled back
                </div>
              )}
            </div>

            <div className="mt-6 grid gap-5">
              {/* Answer */}
              <div className={`${cardInner} p-5`}>
                <div className="text-sm font-semibold text-white/85 mb-2">Answer</div>

                <div className="text-white/90 leading-relaxed text-[15px]">
                  {summary || <span className="text-white/60">(No summary returned — try another query.)</span>}
                </div>

                {keyChips.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {keyChips.map(({ k, v }) => (
                      <span
                        key={k}
                        className="text-xs px-3 py-1 rounded-full border border-white/10 bg-white/[0.04] text-white/75"
                      >
                        <b className="text-white/90">{k}</b>: {v === null || v === undefined ? "" : String(v)}
                      </span>
                    ))}
                  </div>
                )}

                {insightLines.length > 0 && (
                  <div className="mt-4 space-y-2">
                    {insightLines.map((line, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-3 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3"
                      >
                        <span className="mt-1.5 h-2 w-2 rounded-full bg-[#0A84FF] shrink-0" />
                        <span className="text-white/80 text-sm leading-relaxed">{line}</span>
                      </div>
                    ))}
                  </div>
                )}

                <details className="mt-4">
                  <summary className="cursor-pointer text-sm text-white/55 hover:text-white/75 transition">
                    View full answer (raw)
                  </summary>
                  <div className="mt-3 prose prose-invert max-w-none prose-p:text-white/75 prose-strong:text-white">
                    <ReactMarkdown>{resp.answer}</ReactMarkdown>
                  </div>
                </details>

                {resp?.warnings?.length ? (
                  <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] p-3 text-xs text-white/65">
                    <div className="text-white/80 font-semibold mb-1">Notes</div>
                    <ul className="list-disc pl-5 space-y-1">
                      {resp.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>

              {/* SQL + Preview */}
              {includeSql && (
                <>
                  <div className="h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-white/85">SQL & Preview</div>
                    <div className="flex gap-2">
                      <button onClick={() => copyToClipboard(resp.sql)} className={softBtn} disabled={!resp.sql}>
                        Copy SQL
                      </button>
                      <button onClick={exportCSV} className={softBtn} disabled={!resp.rows_preview?.length}>
                        Export CSV
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                    {[
                      ["mode", resp.mode],
                      ["kind", resp.kind],
                      ["latency", `${resp.latency_ms} ms`],
                    ].map(([k, v]) => (
                      <div key={k} className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-white/75">
                        <div className="text-white/45 uppercase tracking-wider text-[10px]">{k}</div>
                        <div className="mt-1 text-white/90 font-semibold">{v}</div>
                      </div>
                    ))}
                  </div>

                  <pre className="overflow-auto rounded-3xl bg-black/45 text-white/85 p-5 text-xs border border-white/10 whitespace-pre-wrap">
{prettySql(resp.sql || "")}
                  </pre>

                  <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-white/85">Preview</div>
                      <div className="text-xs text-white/50 mt-1">
                        Showing <span className="text-white/75 font-semibold">{showingCount}</span> of{" "}
                        <span className="text-white/75 font-semibold">{totalCount}</span> rows
                      </div>
                    </div>

                    <div className="w-full md:w-80">
                      <input
                        value={previewQuery}
                        onChange={(e) => setPreviewQuery(e.target.value)}
                        placeholder="Search in preview…"
                        className="w-full rounded-2xl border border-white/10 bg-black/35 px-4 py-3 text-sm text-white placeholder:text-white/35 focus:outline-none focus:ring-4 focus:ring-[#0A84FF]/15 focus:border-[#0A84FF]/40"
                      />
                    </div>
                  </div>

                  {filteredRows.length ? (
                    <div className="overflow-auto rounded-3xl border border-white/10 bg-white/[0.02]">
                      <table className="w-full text-xs">
                        <thead className="bg-white/[0.04] sticky top-0">
                          <tr>
                            {columns.map((c) => (
                              <th key={c} className="text-left p-4 border-b border-white/10 text-white/75">
                                {c}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {filteredRows.map((row, idx) => (
                            <tr key={idx} className="odd:bg-transparent even:bg-white/[0.02]">
                              {columns.map((c) => (
                                <td key={c} className="p-4 border-b border-white/10 text-white/75">
                                  {row[c] === null || row[c] === undefined ? "" : String(row[c])}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-xs text-white/45">No matching rows for that search.</div>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {/* Examples */}
        <div className={`mt-6 ${card} p-7`}>
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold tracking-tight">Try examples</h2>
              <p className="text-sm text-white/55 mt-1">Click any prompt to run it instantly.</p>
            </div>
            <div className="text-xs text-white/45">
              Tip: switch to <b>sandbox</b> for update/delete simulations.
            </div>
          </div>

          <div className="mt-5 grid gap-5">
            {examples.map((group) => (
              <div key={group.category}>
                <div className="text-sm font-semibold text-white/85">{group.category}</div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {(group.items || []).map((q) => (
                    <button
                      key={q}
                      onClick={() => {
                        setQuestion(q);
                        ask(false, q);
                      }}
                      className="px-4 py-2 rounded-full border border-white/10 bg-white/[0.04] hover:bg-white/[0.08] active:bg-white/[0.10] transition text-sm text-white/80"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            {!examples.length && <div className="text-white/45 text-sm">Loading examples…</div>}
          </div>
        </div>

        {/* Schema modal */}
        {showSchema && (
          <div className="fixed inset-0 bg-black/55 backdrop-blur-sm flex items-center justify-center p-4 z-50">
            <div className="w-full max-w-3xl rounded-3xl border border-white/12 bg-black/55 backdrop-blur-2xl shadow-[0_40px_120px_-60px_rgba(0,0,0,0.95)] p-6">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold tracking-tight">Schema</h2>
                <button onClick={() => setShowSchema(false)} className={softBtn}>
                  Close
                </button>
              </div>

              <div className="mt-4 max-h-[60vh] overflow-auto rounded-3xl border border-white/10 bg-white/[0.03]">
                <table className="w-full text-sm">
                  <thead className="bg-white/[0.04] sticky top-0">
                    <tr>
                      <th className="text-left p-4 border-b border-white/10 text-white/80">Table</th>
                      <th className="text-left p-4 border-b border-white/10 text-white/80">Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {schema.map((t) => (
                      <tr key={t.table_name} className="odd:bg-transparent even:bg-white/[0.02]">
                        <td className="p-4 border-b border-white/10 font-medium text-white/90">
                          {t.table_name}
                        </td>
                        <td className="p-4 border-b border-white/10 text-white/65">
                          {t.description}
                        </td>
                      </tr>
                    ))}
                    {!schema.length && (
                      <tr>
                        <td className="p-4 text-white/50" colSpan={2}>
                          No schema loaded (check TABLE_DESCRIPTIONS_PATH).
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              <p className="text-xs text-white/45 mt-4">
                Public is safe (SELECT-only). Sandbox simulates DML and always rolls back.
              </p>
            </div>
          </div>
        )}

        {showInfo && (
  <div className="fixed inset-0 bg-black/55 backdrop-blur-sm flex items-center justify-center p-4 z-50">
    <div className="w-full max-w-3xl rounded-3xl border border-white/12 bg-black/55 backdrop-blur-2xl shadow-[0_40px_120px_-60px_rgba(0,0,0,0.95)] p-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Welcome to AskDB</h2>
          <p className="text-sm text-white/60 mt-1">
            Ask questions in plain English. AskDB turns them into SQL and shows you the result — fast, transparent, and safe.
          </p>
        </div>
        <button onClick={() => setShowInfo(false)} className={softBtn}>
          Close
        </button>
      </div>

      <div className="mt-5 max-h-[70vh] overflow-auto pr-1">
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
          <div className="text-white/85 font-semibold">What can I do here?</div>
          <div className="text-sm text-white/65 mt-2 leading-relaxed">
            Think of this like <span className="text-white/85 font-semibold">ChatGPT for your database</span>.
            You can ask business questions like “Who are my top customers?” or “What are sales by product line?” —
            and AskDB will generate SQL and show the answer with a preview table.
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
            <div className="text-white/85 font-semibold">Mode 1: Public (Safe)</div>
            <div className="text-sm text-white/65 mt-2">
              Read-only analytics. This mode only runs <span className="text-white/85 font-semibold">SELECT</span> queries.
              Joins and charts-style queries are allowed.
            </div>
            <div className="mt-3 text-xs text-white/55">
              Try: “Top 10 customers by total payments”
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
            <div className="text-white/85 font-semibold">Mode 2: Sandbox (What-If)</div>
            <div className="text-sm text-white/65 mt-2">
              Want to try an update or delete? Sandbox lets you experiment — but
              <span className="text-white/85 font-semibold"> nothing is permanently saved</span>.
              Every change is automatically rolled back.
            </div>
            <div className="mt-3 text-xs text-white/55">
              Try: “Delete customer 112” → shows rollback
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
          <div className="text-white/85 font-semibold">Bring Your Own Database (V2)</div>
          <div className="text-sm text-white/65 mt-2 leading-relaxed">
            You can connect your own database using the <span className="text-white/85 font-semibold">Connect DB</span> button.
            If you don’t have a schema CSV ready, AskDB can auto-detect your schema.
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
          <div className="text-white/85 font-semibold">Quick start (copy/paste)</div>
          <div className="mt-3 flex flex-wrap gap-2">
            {[
              "Top 3 customers by total payments",
              "Total sales by product line",
              "Customers with orders but no payments",
              "In sandbox mode, delete customer 112 safely",
            ].map((q) => (
              <button
                key={q}
                onClick={() => {
                  setQuestion(q);
                  setShowInfo(false);
                  ask(false, q);
                }}
                className="px-4 py-2 rounded-full border border-white/10 bg-white/[0.04] hover:bg-white/[0.08] active:bg-white/[0.10] transition text-sm text-white/80"
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        <div className="text-xs text-white/45">
          Tip: Turn on “Show SQL + preview” to see exactly what query was executed.
        </div>
      </div>
    </div>
  </div>
)}




        {/* Connect DB modal */}
        {showConnect && (
          <div className="fixed inset-0 bg-black/55 backdrop-blur-sm flex items-center justify-center p-4 z-50">
            <div className="w-full max-w-3xl rounded-3xl border border-white/12 bg-black/55 backdrop-blur-2xl shadow-[0_40px_120px_-60px_rgba(0,0,0,0.95)] p-6">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold tracking-tight">Connect your database</h2>
                  <p className="text-sm text-white/60 mt-1">
                    Paste a connection string and optionally provide a schema CSV. If CSV is missing/invalid, AskDB will auto-generate schema from the database.
                  </p>
                </div>
                <button onClick={() => setShowConnect(false)} className={softBtn}>
                  Close
                </button>
              </div>

              <div className="mt-5 grid gap-4">
                <div className="grid gap-2">
                  <div className="text-sm font-semibold text-white/85">Database URL</div>
                  <input
                    value={dbUrl}
                    onChange={(e) => setDbUrl(e.target.value)}
                    placeholder="postgresql+psycopg2://user:password@host:5432/db?sslmode=require"
                    className={inputBase}
                  />
                  {dbUrl ? (
                    <div className="text-xs text-white/45">
                      Preview: <span className="text-white/70">{maskDbUrl(dbUrl)}</span>
                    </div>
                  ) : null}
                </div>

                <div className="grid gap-2">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-white/85">Schema CSV (optional)</div>
                    <label className={`${softBtn} cursor-pointer`}>
                      Upload CSV
                      <input
                        type="file"
                        accept=".csv,text/csv"
                        className="hidden"
                        onChange={(e) => onUploadCsvFile(e.target.files?.[0])}
                      />
                    </label>
                  </div>

                  <textarea
                    value={schemaCsv}
                    onChange={(e) => setSchemaCsv(e.target.value)}
                    rows={6}
                    placeholder={`table_name,description
customers,Contains customer info
orders,Order headers
orderdetails,Line items
payments,Payments per customer`}
                    className={inputBase}
                  />

                  <div className="text-xs text-white/45">
                    Supported formats: we auto-detect header variants. If invalid, we fallback to DB inspection automatically.
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <button onClick={connectDb} disabled={connecting || !dbUrl.trim()} className={primaryBtn}>
                    {connecting ? "Connecting…" : "Connect"}
                  </button>

                  <button
                    onClick={disconnectDb}
                    className={dangerBtn}
                    disabled={!conn.connected || connecting}
                    title={!conn.connected ? "No BYODB connection to disconnect" : "Disconnect and go back to demo DB"}
                  >
                    {connecting ? "…" : "Disconnect"}
                  </button>

                  <button
                    onClick={() => {
                      setDbUrl("");
                      setSchemaCsv("");
                      showToast("Cleared");
                    }}
                    className={softBtn}
                    type="button"
                  >
                    Clear
                  </button>

                  <div className="ml-auto text-xs text-white/50">
                    Session: <span className="text-white/70">{sessionId}</span>
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3 text-xs text-white/65">
                  <div className="text-white/80 font-semibold mb-1">Notes</div>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Public mode is safest for BYODB (SELECT-only).</li>
                    <li>Sandbox mode simulates DML and rolls back—still recommended to gate behind a demo key in production.</li>
                    <li>When switching databases, it’s best to reset the session to avoid cross-DB context confusion.</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="mt-10 text-center text-xs text-white/35">
          AskDB • Apple-inspired UI • V2 BYODB (Connect/Disconnect)
        </div>
      </div>
    </div>
  );
}
