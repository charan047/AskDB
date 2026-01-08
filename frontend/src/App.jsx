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
    .replace(/^\s*[*\-\u2022]\s*/, "") // remove bullets
    .replace(/\*\*/g, "")              // remove markdown bold markers
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

// ---------- main ----------
export default function App() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("public");
  const [includeSql, setIncludeSql] = useState(true);
  const [loading, setLoading] = useState(false);

  const [about, setAbout] = useState(null);
  const [examples, setExamples] = useState([]);
  const [schema, setSchema] = useState([]);
  const [showSchema, setShowSchema] = useState(false);

  const [resp, setResp] = useState(null);
  const [error, setError] = useState("");

  // Step 2: loading stepper
  const [phase, setPhase] = useState("idle"); // idle | sql | db | answer

  // Step 3: preview search
  const [previewQuery, setPreviewQuery] = useState("");

  // Step 1: toast
  const [toast, setToast] = useState({ open: false, message: "" });
  const toastTimerRef = useRef(null);

  const resultsRef = useRef(null);

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
        // backend not ready; ignore
      }
    })();
  }, []);

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

    // Premium stepper simulation (UI only)
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

  // ---------- Dark premium tokens (single teal accent) ----------
  const card =
    "rounded-2xl border border-white/10 bg-white/[0.035] backdrop-blur shadow-[0_20px_60px_-35px_rgba(0,0,0,0.85)]";
  const softBtn =
    "px-4 py-2 rounded-xl border border-white/10 bg-white/[0.04] hover:bg-white/[0.07] transition text-white/80";
  const primaryBtn =
    "px-5 py-2 rounded-xl bg-teal-500 text-slate-950 font-semibold hover:bg-teal-400 disabled:opacity-60 transition";
  const chip =
    "inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs text-white/70";

  const inputBase =
    "w-full rounded-2xl border border-white/10 bg-black/35 p-3 text-white placeholder:text-white/35 focus:outline-none focus:ring-4 focus:ring-teal-500/10 focus:border-teal-400/40";

  const miniPill =
    "px-2 py-1 rounded-lg border border-white/10 bg-black/30 text-xs text-white/70";

  // ---------- Step 4: executive answer formatting ----------
  const rawAnswer = String(resp?.answer || "");
  const lines = rawAnswer.split("\n").map(normalizeLine).filter(Boolean);

  const summary = lines[0] || "";
  const insightLines = lines.slice(1, 10);

  // chips from first preview row (if exists)
  const firstRow = resp?.rows_preview?.[0] || null;
  const chipKeys = firstRow ? Object.keys(firstRow).slice(0, 6) : [];
  const keyChips = firstRow
    ? chipKeys.map((k) => ({ k, v: firstRow[k] }))
    : [];

  // ---------- Step 3: filter preview rows ----------
  const previewRows = resp?.rows_preview || [];
  const filteredRows = previewRows.filter((r) => lineSearchMatch(r, previewQuery));
  const columns = resp?.columns?.length ? resp.columns : (previewRows[0] ? Object.keys(previewRows[0]) : []);
  const showingCount = filteredRows.length;
  const totalCount = resp?.rows_returned ?? previewRows.length;

  return (
    <div className="min-h-screen text-white">
      {/* Background: clean & premium, single teal glow */}
      <div className="fixed inset-0 -z-10 bg-[#070A0F]" />
      <div className="fixed inset-0 -z-10 opacity-70 bg-[radial-gradient(circle_at_18%_18%,rgba(0, 0, 0, 0.2),transparent_45%)]" />
      <div className="fixed inset-0 -z-10 opacity-25 bg-[linear-gradient(to_bottom,rgba(255,255,255,0.06),transparent_35%,rgba(255,255,255,0.02))]" />

      {/* Toast */}
      <div
        className={`fixed top-4 right-4 z-50 transition ${
          toast.open ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        <div className="rounded-xl border border-white/10 bg-[#0B0F14] px-4 py-2 text-sm text-white/85 shadow-[0_20px_60px_-35px_rgba(0,0,0,0.9)]">
          <span className="inline-block w-2 h-2 rounded-full bg-teal-400 mr-2 align-middle" />
          {toast.message}
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 py-10">
        {/* Header */}
        <div className={`${card} p-6`}>
          <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-5">
            <div>
              <div className="flex flex-wrap gap-2">
                <span className={chip}>AskDB</span>
                <span className={chip}>CRM Analytics Copilot</span>
                <span className={chip}>Rollback Sandbox</span>
              </div>

              <h1 className="text-2xl md:text-3xl font-semibold mt-3 tracking-tight">
                Natural language → SQL → Business insight
              </h1>

              <p className="text-white/60 mt-2">
                Public mode is safe analytics (SELECT-only). Sandbox mode simulates updates/deletes and always rolls back.
              </p>

              <div className="mt-4 text-sm text-white/60">
                {about ? (
                  <>
                    <div>
                      <span className="font-semibold text-white/85">Dataset:</span>{" "}
                      {about.dataset?.name}
                    </div>
                    <div className="mt-1">
                      <span className="font-semibold text-white/85">Entities:</span>{" "}
                      {(about.dataset?.entities || []).join(", ")}
                    </div>
                  </>
                ) : (
                  <div className="text-white/40">Connecting to backend…</div>
                )}
              </div>

              <p className="text-xs text-white/35 mt-3">Session: {sessionId}</p>
            </div>

            <div className="flex gap-2">
              <button onClick={() => setShowSchema(true)} className={softBtn}>
                View schema
              </button>
              <button onClick={() => ask(true)} className={softBtn} title="Clear conversation context">
                Reset session
              </button>
            </div>
          </div>
        </div>

        {/* Ask */}
        <div className={`mt-6 ${card} p-6`}>
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold">Ask a question</h2>
              <p className="text-sm text-white/55 mt-1">
                Use public for analytics. Use sandbox to simulate updates/deletes safely.
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <span className={miniPill}>public = SELECT-only</span>
              <span className={miniPill}>sandbox = rollback</span>
            </div>
          </div>

          <div className="mt-4 grid gap-3">
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
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  className="rounded-xl border border-white/10 px-3 py-2 bg-black/35 text-white"
                >
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

            {/* Step 2: Loading stepper */}
            {loading && (
              <div className="mt-3">
                <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className={`h-full bg-teal-400/80 transition-all duration-500 ${
                      phase === "sql" ? "w-1/4" : phase === "db" ? "w-2/4" : phase === "answer" ? "w-3/4" : "w-0"
                    }`}
                  />
                </div>

                <div className="mt-2 flex flex-wrap gap-2 text-xs">
                  <span className={`px-2 py-1 rounded-lg border ${
                    phase === "sql" ? "border-teal-400/30 bg-teal-500/10 text-teal-200" : "border-white/10 bg-white/[0.03] text-white/50"
                  }`}>
                    1) Generating SQL
                  </span>
                  <span className={`px-2 py-1 rounded-lg border ${
                    phase === "db" ? "border-teal-400/30 bg-teal-500/10 text-teal-200" : "border-white/10 bg-white/[0.03] text-white/50"
                  }`}>
                    2) Querying DB
                  </span>
                  <span className={`px-2 py-1 rounded-lg border ${
                    phase === "answer" ? "border-teal-400/30 bg-teal-500/10 text-teal-200" : "border-white/10 bg-white/[0.03] text-white/50"
                  }`}>
                    3) Writing answer
                  </span>
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
          <div ref={resultsRef} className={`mt-6 ${card} p-6`}>
            <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">Results</h2>
                <p className="text-sm text-white/55 mt-1">
                  Executive summary + SQL transparency + preview table.
                </p>
              </div>

              {resp.mode === "sandbox" && resp.rolled_back && (
                <div className="shrink-0 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-white/75">
                  ✅ Sandbox simulation — rolled back
                </div>
              )}
            </div>

            {/* Step 4: Executive answer */}
            <div className="mt-4">
              <div className="text-sm font-semibold text-white/85 mb-2">Answer</div>

              {/* Summary */}
              <div className="text-white/90 leading-relaxed">
                {summary || (
                  <span className="text-white/60">
                    (No summary returned — try another query.)
                  </span>
                )}
              </div>

              {/* Key fields chips from first row */}
              {keyChips.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {keyChips.map(({ k, v }) => (
                    <span key={k} className="text-xs px-2 py-1 rounded-lg border border-white/10 bg-black/30 text-white/75">
                      <b className="text-white/85">{k}</b>: {v === null || v === undefined ? "" : String(v)}
                    </span>
                  ))}
                </div>
              )}

              {/* Insights as clean cards */}
              {insightLines.length > 0 && (
                <div className="mt-4 space-y-2">
                  {insightLines.map((line, i) => (
                    <div
                      key={i}
                      className="flex items-start gap-2 rounded-xl border border-white/10 bg-black/25 px-3 py-2"
                    >
                      <span className="mt-1 h-2 w-2 rounded-full bg-teal-400/80 shrink-0" />
                      <span className="text-white/80 text-sm leading-relaxed">{line}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Optional: keep markdown for long/complex answers (collapsed) */}
              <details className="mt-4">
                <summary className="cursor-pointer text-sm text-white/60 hover:text-white/75 transition">
                  View full answer (raw)
                </summary>
                <div className="mt-3 prose prose-invert max-w-none prose-p:text-white/75 prose-strong:text-white">
                  <ReactMarkdown>{resp.answer}</ReactMarkdown>
                </div>
              </details>
            </div>

            {/* SQL & Preview */}
            {includeSql && (
              <>
                <div className="my-6 h-px bg-white/10" />

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

                <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-2 text-xs text-white/70">
                  <div className="rounded-xl border border-white/10 bg-black/30 px-3 py-2">
                    <b className="text-white/80">mode</b>: {resp.mode}
                  </div>
                  <div className="rounded-xl border border-white/10 bg-black/30 px-3 py-2">
                    <b className="text-white/80">kind</b>: {resp.kind}
                  </div>
                  <div className="rounded-xl border border-white/10 bg-black/30 px-3 py-2">
                    <b className="text-white/80">latency</b>: {resp.latency_ms} ms
                  </div>
                </div>

                <pre className="mt-4 overflow-auto rounded-2xl bg-black/45 text-white/85 p-4 text-xs border border-white/10 whitespace-pre-wrap">
{prettySql(resp.sql || "")}
                </pre>

                {/* Step 3: search + row count */}
                <div className="mt-4 flex flex-col md:flex-row md:items-end md:justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-white/85">Preview</div>
                    <div className="text-xs text-white/50 mt-1">
                      Showing <span className="text-white/75 font-semibold">{showingCount}</span>{" "}
                      of <span className="text-white/75 font-semibold">{totalCount}</span> rows
                    </div>
                  </div>

                  <div className="w-full md:w-72">
                    <input
                      value={previewQuery}
                      onChange={(e) => setPreviewQuery(e.target.value)}
                      placeholder="Search in preview…"
                      className="w-full rounded-xl border border-white/10 bg-black/35 px-3 py-2 text-sm text-white placeholder:text-white/35 focus:outline-none focus:ring-4 focus:ring-teal-500/10 focus:border-teal-400/40"
                    />
                  </div>
                </div>

                {filteredRows.length ? (
                  <div className="mt-3 overflow-auto border border-white/10 rounded-2xl bg-black/25">
                    <table className="w-full text-xs">
                      <thead className="bg-white/[0.04] sticky top-0">
                        <tr>
                          {columns.map((c) => (
                            <th key={c} className="text-left p-3 border-b border-white/10 text-white/75">
                              {c}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {filteredRows.map((row, idx) => (
                          <tr key={idx} className="odd:bg-transparent even:bg-white/[0.02]">
                            {columns.map((c) => (
                              <td key={c} className="p-3 border-b border-white/10 text-white/75">
                                {row[c] === null || row[c] === undefined ? "" : String(row[c])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="mt-3 text-xs text-white/45">
                    No matching rows for that search.
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* Examples */}
        <div className={`mt-6 ${card} p-6`}>
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold">Try examples</h2>
              <p className="text-sm text-white/55 mt-1">Click any example to auto-run it.</p>
            </div>
            <div className="text-xs text-white/40">
              Tip: switch to <b>sandbox</b> for update/delete simulations.
            </div>
          </div>

          <div className="mt-4 grid gap-4">
            {examples.map((group) => (
              <div key={group.category}>
                <div className="text-sm font-semibold text-white/85">{group.category}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {(group.items || []).map((q) => (
                    <button
                      key={q}
                      onClick={() => {
                        setQuestion(q);
                        ask(false, q);
                      }}
                      className="px-3 py-2 rounded-xl border border-white/10 bg-white/[0.04] hover:bg-white/[0.07] transition text-sm text-white/75"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            {!examples.length && <div className="text-white/40 text-sm">Loading examples…</div>}
          </div>
        </div>

        {/* Schema modal */}
        {showSchema && (
          <div className="fixed inset-0 bg-black/60 flex items-center justify-center p-4 z-50">
            <div className="w-full max-w-3xl rounded-2xl border border-white/10 bg-[#0B0F14] shadow-[0_30px_100px_-50px_rgba(0,0,0,0.9)] p-6">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Schema</h2>
                <button onClick={() => setShowSchema(false)} className={softBtn}>
                  Close
                </button>
              </div>

              <div className="mt-4 max-h-[60vh] overflow-auto border border-white/10 rounded-2xl bg-black/30">
                <table className="w-full text-sm">
                  <thead className="bg-white/[0.04] sticky top-0">
                    <tr>
                      <th className="text-left p-3 border-b border-white/10 text-white/80">Table</th>
                      <th className="text-left p-3 border-b border-white/10 text-white/80">Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {schema.map((t) => (
                      <tr key={t.table_name} className="odd:bg-transparent even:bg-white/[0.02]">
                        <td className="p-3 border-b border-white/10 font-medium text-white/85">{t.table_name}</td>
                        <td className="p-3 border-b border-white/10 text-white/65">{t.description}</td>
                      </tr>
                    ))}
                    {!schema.length && (
                      <tr>
                        <td className="p-3 text-white/45" colSpan={2}>
                          No schema loaded (check TABLE_DESCRIPTIONS_PATH).
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              <p className="text-xs text-white/45 mt-3">
                Public is safe (SELECT-only). Sandbox simulates DML and always rolls back.
              </p>
            </div>
          </div>
        )}

        <div className="mt-8 text-center text-xs text-white/30">
          AskDB • clean dark premium UI
        </div>
      </div>
    </div>
  );
}
