"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Page {
  page_number: number;
  text: string;
  method: "direct" | "ocr";
  character_count: number;
}
interface PdfResult {
  kind: "pdf";
  filename: string;
  total_pages: number;
  total_characters: number;
  extraction_time_seconds: number;
  pages: Page[];
}
interface ImageResult { kind: "image"; filename: string; text: string; }
type ExtractResult = PdfResult | ImageResult;
type Status = "idle" | "loading" | "success" | "error";
type AppMode = "single" | "compare" | "merge" | "spreadsheet";
interface MatchRef { pageNum: number; localIdx: number; }
interface MergeSection { filename: string; kind: "pdf" | "image"; pages: Page[]; }

// ─── Constants ────────────────────────────────────────────────────────────────

const PDF_EXTS = [".pdf"];
const IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"];
const ACCEPTED_EXTS = [...PDF_EXTS, ...IMAGE_EXTS];
const WORDS_PER_MINUTE = 200;
const STOP_WORDS = new Set([
  "the","a","an","and","or","but","in","on","at","to","for","of","with","is","are",
  "was","were","be","been","have","has","had","do","does","did","will","would",
  "could","should","this","that","these","those","it","its","i","you","he","she",
  "we","they","me","him","her","us","them","my","your","his","our","their","from",
  "by","as","if","not","no","so","then","than","what","which","who","all","any",
  "also","into","about","up","out","just","more","one","two","can","new",
]);

// ─── Context ──────────────────────────────────────────────────────────────────

const DarkCtx = createContext(false);

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getExtension(name: string) {
  const dot = name.lastIndexOf(".");
  return dot !== -1 ? name.slice(dot).toLowerCase() : "";
}
function fmt(n: number) { return n.toLocaleString(); }
function escapeRegex(s: string) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

function buildRegex(q: string, cs: boolean, ww: boolean): RegExp | null {
  if (!q.trim()) return null;
  try {
    let p = escapeRegex(q);
    if (ww) p = `\\b${p}\\b`;
    return new RegExp(p, cs ? "g" : "gi");
  } catch { return null; }
}
function countMatches(text: string, regex: RegExp): number {
  regex.lastIndex = 0;
  return (text.match(regex) ?? []).length;
}
function buildAllMatches(pages: Page[], regex: RegExp | null, getEff: (p: Page) => string): MatchRef[] {
  if (!regex) return [];
  const refs: MatchRef[] = [];
  for (const page of pages) {
    const n = countMatches(getEff(page), regex);
    for (let i = 0; i < n; i++) refs.push({ pageNum: page.page_number, localIdx: i });
  }
  return refs;
}
function applyRedactStr(text: string, terms: string[]): string {
  let t = text;
  for (const term of terms) {
    if (term.trim()) t = t.replace(new RegExp(escapeRegex(term), "gi"), "█".repeat(term.length));
  }
  return t;
}
function highlightText(text: string, regex: RegExp | null, curIdx: number): React.ReactNode {
  if (!regex || !text) return text;
  const parts: React.ReactNode[] = [];
  let last = 0, mc = 0;
  let m: RegExpExecArray | null;
  regex.lastIndex = 0;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    parts.push(
      <mark key={mc} className={mc === curIdx
        ? "bg-orange-400 text-white rounded px-0.5"
        : "bg-yellow-200 text-yellow-900 rounded px-0.5"}>
        {m[0]}
      </mark>
    );
    last = m.index + m[0].length; mc++;
    if (m[0].length === 0) regex.lastIndex++;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.length ? parts : text;
}
function detectLanguage(text: string): string {
  if (!text.trim()) return "Unknown";
  const scripts: [string, RegExp][] = [
    ["Hindi", /[\u0900-\u097F]/], ["Telugu", /[\u0C00-\u0C7F]/],
    ["Tamil", /[\u0B80-\u0BFF]/], ["Kannada", /[\u0C80-\u0CFF]/],
    ["Malayalam", /[\u0D00-\u0D7F]/], ["Arabic", /[\u0600-\u06FF]/],
    ["Chinese", /[\u4E00-\u9FFF]/], ["Japanese", /[\u3040-\u30FF]/],
    ["Korean", /[\uAC00-\uD7AF]/], ["Russian", /[\u0400-\u04FF]/],
    ["Greek", /[\u0370-\u03FF]/],
  ];
  let maxC = 0, detected = "";
  for (const [lang, rx] of scripts) {
    const c = (text.match(new RegExp(rx.source, "g")) ?? []).length;
    if (c > maxC) { maxC = c; detected = lang; }
  }
  if (maxC >= 10) return detected;
  return (text.match(/[a-zA-Z]/g) ?? []).length >= 10 ? "English" : "Unknown";
}
function detectTable(text: string): boolean {
  const lines = text.split("\n").filter((l) => l.trim());
  if (lines.length < 3) return false;
  return lines.filter((l) => (l.match(/\t/g) ?? []).length >= 2).length >= 3
    || lines.filter((l) => (l.match(/\|/g) ?? []).length >= 2).length >= 3
    || lines.filter((l) => /\S {3,}\S/.test(l)).length >= Math.ceil(lines.length * 0.5);
}
function getWordFreq(text: string, limit = 20): [string, number][] {
  const words = text.toLowerCase().match(/\b[a-z][a-z'-]{2,}\b/g) ?? [];
  const freq: Record<string, number> = {};
  for (const w of words) if (!STOP_WORDS.has(w)) freq[w] = (freq[w] ?? 0) + 1;
  return Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, limit);
}
function buildTxt(pages: Page[], getEff: (p: Page) => string) {
  return pages.map((p) => `--- Page ${p.page_number} ---\n${getEff(p)}`).join("\n\n");
}
function buildJson(result: ExtractResult, pages: Page[], getEff: (p: Page) => string) {
  if (result.kind === "image") return JSON.stringify(result, null, 2);
  return JSON.stringify({ ...result, pages: pages.map((p) => ({ ...p, text: getEff(p) })) }, null, 2);
}
function buildMd(filename: string, pages: Page[], getEff: (p: Page) => string) {
  return `# ${filename}\n\n---\n\n` + pages.map((p) =>
    `## Page ${p.page_number}\n\n${getEff(p).trim() || "_No text found._"}`
  ).join("\n\n---\n\n");
}
function buildMergeTxt(sections: MergeSection[]) {
  return sections.map((s) =>
    `${"═".repeat(4)} ${s.filename} ${"═".repeat(4)}\n\n` +
    s.pages.map((p) => `--- Page ${p.page_number} ---\n${p.text}`).join("\n\n")
  ).join("\n\n\n");
}
function buildSummaryHtml(result: PdfResult, wordCount: number, lang: string) {
  const rows = result.pages.map((p) =>
    `<tr><td>${p.page_number}</td><td>${p.method === "ocr" ? "OCR" : "Direct"}</td><td>${fmt(p.character_count)}</td></tr>`
  ).join("");
  return `<!DOCTYPE html><html><head><title>Summary – ${result.filename}</title><style>
body{font-family:Georgia,serif;max-width:700px;margin:2rem auto;color:#1a202c;line-height:1.6}
h1{font-size:1.4rem;border-bottom:2px solid #e2e8f0;padding-bottom:.5rem;margin-bottom:1.5rem}
h2{font-size:1rem;margin-top:1.5rem}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1.5rem 0}
.stat{background:#f7fafc;border:1px solid #e2e8f0;border-radius:6px;padding:.75rem;text-align:center}
.stat .v{font-size:1.4rem;font-weight:700;color:#2b6cb0}.stat .l{font-size:.7rem;color:#718096;text-transform:uppercase;letter-spacing:.06em;margin-top:.2rem}
table{width:100%;border-collapse:collapse}th,td{padding:.4rem .6rem;border-bottom:1px solid #e2e8f0;font-size:.85rem;text-align:left}
th{background:#f7fafc;font-weight:600}.badge{background:#ebf8ff;color:#2b6cb0;border-radius:999px;padding:.1rem .6rem;font-size:.8rem}
@media print{@page{margin:2cm}}</style></head><body>
<h1>Extraction Summary</h1>
<p><strong>File:</strong> ${result.filename}</p>
<p><strong>Generated:</strong> ${new Date().toLocaleDateString()}</p>
<p><strong>Language:</strong> <span class="badge">${lang}</span></p>
<div class="grid">
  <div class="stat"><div class="v">${result.total_pages}</div><div class="l">Pages</div></div>
  <div class="stat"><div class="v">${fmt(wordCount)}</div><div class="l">Words</div></div>
  <div class="stat"><div class="v">${fmt(result.total_characters)}</div><div class="l">Characters</div></div>
  <div class="stat"><div class="v">${result.extraction_time_seconds}s</div><div class="l">Extract Time</div></div>
  <div class="stat"><div class="v">~${Math.ceil(wordCount / 200)} min</div><div class="l">Reading Time</div></div>
  <div class="stat"><div class="v">${result.pages.filter(p => p.method === "ocr").length}</div><div class="l">OCR Pages</div></div>
</div>
<h2>Per-Page Details</h2>
<table><tr><th>Page</th><th>Method</th><th>Characters</th></tr>${rows}</table>
</body></html>`;
}
function triggerDownload(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function fetchExtract(file: File): Promise<ExtractResult> {
  const ext = getExtension(file.name);
  const endpoint = PDF_EXTS.includes(ext)
    ? `${API_BASE}/extract-text`
    : `${API_BASE}/extract-text-image`;
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(endpoint, { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail ?? "Extraction failed");
  return PDF_EXTS.includes(ext) ? { kind: "pdf", ...data } : { kind: "image", ...data };
}

// ─── StatsPanel ───────────────────────────────────────────────────────────────

function StatsPanel({ pages, text }: { pages: Page[]; text: string }) {
  const dark = useContext(DarkCtx);
  const topWords = useMemo(() => getWordFreq(text), [text]);
  const maxCount = topWords[0]?.[1] ?? 1;
  const wordCounts = pages.map((p) => p.text.trim() ? p.text.trim().split(/\s+/).length : 0);
  const avgWords = pages.length ? Math.round(wordCounts.reduce((a, b) => a + b, 0) / pages.length) : 0;
  const sorted = [...pages].sort((a, b) => a.character_count - b.character_count);

  return (
    <div className={`rounded-2xl border p-6 space-y-5 ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
      <h3 className={`text-sm font-semibold ${dark ? "text-gray-300" : "text-gray-700"}`}>Text Statistics</h3>
      <div className="grid grid-cols-3 gap-4">
        {[
          { l: "Avg words / page", v: fmt(avgWords) },
          { l: "Shortest page", v: sorted[0] ? `Page ${sorted[0].page_number}` : "—" },
          { l: "Longest page", v: sorted[sorted.length - 1] ? `Page ${sorted[sorted.length - 1].page_number}` : "—" },
        ].map(({ l, v }) => (
          <div key={l} className={`rounded-xl p-3 ${dark ? "bg-gray-700/50" : "bg-gray-50"}`}>
            <p className={`text-xs ${dark ? "text-gray-500" : "text-gray-400"}`}>{l}</p>
            <p className={`text-sm font-semibold mt-0.5 ${dark ? "text-white" : "text-gray-900"}`}>{v}</p>
          </div>
        ))}
      </div>
      <div>
        <p className={`text-xs font-semibold uppercase tracking-wide mb-3 ${dark ? "text-gray-500" : "text-gray-400"}`}>Top {topWords.length} Words</p>
        <div className="space-y-1.5">
          {topWords.map(([word, count]) => (
            <div key={word} className="flex items-center gap-2">
              <span className={`w-24 text-xs text-right truncate ${dark ? "text-gray-400" : "text-gray-600"}`}>{word}</span>
              <div className={`flex-1 rounded-full h-2 ${dark ? "bg-gray-700" : "bg-gray-100"}`}>
                <div className="bg-blue-500 h-2 rounded-full transition-all" style={{ width: `${(count / maxCount) * 100}%` }} />
              </div>
              <span className={`w-6 text-right text-xs ${dark ? "text-gray-500" : "text-gray-400"}`}>{count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── RedactPanel ──────────────────────────────────────────────────────────────

function RedactPanel({ terms, onAdd, onRemove }: { terms: string[]; onAdd: (t: string) => void; onRemove: (i: number) => void }) {
  const dark = useContext(DarkCtx);
  const [input, setInput] = useState("");
  function add() { if (input.trim()) { onAdd(input.trim()); setInput(""); } }
  return (
    <div className={`rounded-2xl border p-5 space-y-3 ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
      <h3 className={`text-sm font-semibold ${dark ? "text-gray-300" : "text-gray-700"}`}>Redact Terms</h3>
      <div className="flex gap-2">
        <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && add()}
          placeholder="Word or phrase to redact…"
          className={`flex-1 rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-red-500 ${dark ? "bg-gray-700 border-gray-600 text-gray-100 placeholder-gray-500" : "bg-white border-gray-300 text-gray-900 placeholder-gray-400"}`} />
        <button onClick={add} className="rounded-lg bg-red-500 px-4 py-2 text-sm font-medium text-white hover:bg-red-600 transition-colors">Add</button>
      </div>
      {terms.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {terms.map((t, i) => (
            <span key={i} className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-medium ${dark ? "bg-red-900/40 text-red-300" : "bg-red-50 text-red-700 border border-red-200"}`}>
              {t}<button onClick={() => onRemove(i)} className="ml-1 opacity-60 hover:opacity-100 text-sm">&times;</button>
            </span>
          ))}
        </div>
      )}
      <p className={`text-xs ${dark ? "text-gray-600" : "text-gray-400"}`}>Redacted terms appear as blocks in view and all downloads.</p>
    </div>
  );
}

// ─── ComparePanel ─────────────────────────────────────────────────────────────

function ComparePanel({ label }: { label: string }) {
  const dark = useContext(DarkCtx);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<ExtractResult | null>(null);
  const [err, setErr] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  async function processFile(file: File) {
    if (!ACCEPTED_EXTS.includes(getExtension(file.name))) { setErr("Unsupported file type."); return; }
    setStatus("loading"); setResult(null); setErr("");
    try { setResult(await fetchExtract(file)); setStatus("success"); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "Error"); setStatus("error"); }
  }

  return (
    <div className="flex flex-col">
      <div className={`px-4 py-2.5 rounded-t-2xl border border-b-0 text-xs font-bold uppercase tracking-wider ${dark ? "bg-gray-800 border-gray-700 text-gray-500" : "bg-blue-50 border-blue-200 text-blue-600"}`}>{label}</div>
      <div className={`rounded-b-2xl border overflow-auto max-h-[72vh] ${dark ? "bg-gray-900 border-gray-700" : "bg-white border-gray-200"}`}>
        {(status === "idle" || status === "error") && (
          <div className="flex flex-col items-center justify-center h-48 gap-3 cursor-pointer p-6" onClick={() => inputRef.current?.click()}>
            <input ref={inputRef} type="file" accept={ACCEPTED_EXTS.join(",")} className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) processFile(f); e.target.value = ""; }} />
            <svg className={`w-10 h-10 ${dark ? "text-gray-700" : "text-blue-200"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
            <p className={`text-sm ${dark ? "text-gray-600" : "text-gray-400"}`}>Click to upload</p>
            {status === "error" && <p className="text-xs text-red-500">{err}</p>}
          </div>
        )}
        {status === "loading" && (
          <div className="flex items-center justify-center h-48">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        )}
        {status === "success" && result && (
          <div className="px-6 py-4 space-y-4">
            <p className={`text-xs font-semibold ${dark ? "text-gray-400" : "text-blue-600"}`}>{result.filename}</p>
            {result.kind === "pdf" ? result.pages.map((p) => (
              <div key={p.page_number} className={`rounded-xl border p-4 ${dark ? "bg-gray-800 border-gray-700" : "bg-gray-50 border-gray-200"}`}>
                <p className={`text-xs font-semibold mb-2 ${dark ? "text-gray-500" : "text-gray-400"}`}>Page {p.page_number}</p>
                <p className={`text-sm whitespace-pre-wrap leading-relaxed ${dark ? "text-gray-300" : "text-gray-700"}`}>{p.text || "—"}</p>
              </div>
            )) : (
              <p className={`text-sm whitespace-pre-wrap leading-relaxed ${dark ? "text-gray-300" : "text-gray-700"}`}>{(result as ImageResult).text || "—"}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function Home() {
  const [appMode, setAppMode] = useState<AppMode>("single");
  const [darkMode, setDarkMode] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [fileDragging, setFileDragging] = useState(false);
  const [result, setResult] = useState<ExtractResult | null>(null);
  const [errorMsg, setErrorMsg] = useState("");

  const [pageOrder, setPageOrder] = useState<number[]>([]);
  const dragSrcIdxRef = useRef<number | null>(null);
  const [draggingPageNum, setDraggingPageNum] = useState<number | null>(null);

  const [collapsedPages, setCollapsedPages] = useState<Set<number>>(new Set());
  const [bookmarkedPages, setBookmarkedPages] = useState<Set<number>>(new Set());

  // Search
  const [query, setQuery] = useState("");
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [wholeWord, setWholeWord] = useState(false);
  const [currentMatchIdx, setCurrentMatchIdx] = useState(0);
  const [showReplace, setShowReplace] = useState(false);
  const [replaceWith, setReplaceWith] = useState("");
  const [modifiedTexts, setModifiedTexts] = useState<Map<number, string>>(new Map());
  const searchInputRef = useRef<HTMLInputElement>(null);

  // View options
  const [showPageNav, setShowPageNav] = useState(true);
  const [fromPage, setFromPage] = useState(1);
  const [toPage, setToPage] = useState(1);
  const [showStats, setShowStats] = useState(false);
  const [showRedact, setShowRedact] = useState(false);
  const [redactTerms, setRedactTerms] = useState<string[]>([]);

  // Copy
  const [copied, setCopied] = useState(false);
  const [showGoTop, setShowGoTop] = useState(false);

  // Info panel
  const [showInfo, setShowInfo] = useState(false);
  const infoRef = useRef<HTMLDivElement>(null);

  // Page nav collapse
  const [pageNavExpanded, setPageNavExpanded] = useState(false);
  const PAGE_NAV_LIMIT = 20;

  // Merge mode
  const [mergedSections, setMergedSections] = useState<MergeSection[]>([]);
  const [mergeStatus, setMergeStatus] = useState<Status>("idle");
  const [mergeError, setMergeError] = useState("");
  const mergeInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Spreadsheet mode
  const [ssFields, setSsFields] = useState<string[]>([""]);
  const [ssFiles, setSsFiles] = useState<File[]>([]);
  const [ssStatus, setSsStatus] = useState<Status>("idle");
  const [ssError, setSsError] = useState("");
  const [ssPreview, setSsPreview] = useState<Record<string, string>[] | null>(null);
  const [ssContext, setSsContext] = useState<Record<string, string>[] | null>(null);
  const [ssMethod, setSsMethod] = useState<"auto" | "block">("auto");
  const [ssShowContext, setSsShowContext] = useState(false);
  const [ssDetecting, setSsDetecting] = useState(false);
  const ssInputRef = useRef<HTMLInputElement>(null);

  // Download dropdown
  const [showDownload, setShowDownload] = useState(false);
  const downloadRef = useRef<HTMLDivElement>(null);

  // Effects
  useEffect(() => {
    const saved = localStorage.getItem("darkMode");
    if (saved === "true") setDarkMode(true);
  }, []);
  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    localStorage.setItem("darkMode", String(darkMode));
  }, [darkMode]);
  useEffect(() => {
    const onScroll = () => setShowGoTop(window.scrollY > 400);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => {
    if (result?.kind === "pdf") {
      setPageOrder(result.pages.map((p) => p.page_number));
      setFromPage(1); setToPage(result.total_pages);
      setCollapsedPages(new Set()); setBookmarkedPages(new Set());
      setModifiedTexts(new Map()); setRedactTerms([]);
      setQuery(""); setShowPageNav(true);
    }
  }, [result]);
  useEffect(() => { setCurrentMatchIdx(0); }, [query, caseSensitive, wholeWord, fromPage, toPage]);
  useEffect(() => {
    function h(e: MouseEvent) {
      if (downloadRef.current && !downloadRef.current.contains(e.target as Node)) setShowDownload(false);
      if (infoRef.current && !infoRef.current.contains(e.target as Node)) setShowInfo(false);
    }
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  // Derived
  const getEff = useCallback((page: Page): string => {
    const text = modifiedTexts.get(page.page_number) ?? page.text;
    return applyRedactStr(text, redactTerms);
  }, [modifiedTexts, redactTerms]);

  const orderedPages = useMemo<Page[]>(() => {
    if (!result || result.kind !== "pdf") return [];
    return pageOrder.map((n) => result.pages.find((p) => p.page_number === n)).filter(Boolean) as Page[];
  }, [result, pageOrder]);

  const pagesInRange = useMemo(
    () => orderedPages.filter((p) => p.page_number >= fromPage && p.page_number <= toPage),
    [orderedPages, fromPage, toPage]
  );
  const searchRegex = useMemo(() => buildRegex(query, caseSensitive, wholeWord), [query, caseSensitive, wholeWord]);
  const allMatches = useMemo(() => buildAllMatches(pagesInRange, searchRegex, getEff), [pagesInRange, searchRegex, getEff]);
  const safeMatchIdx = allMatches.length > 0 ? Math.min(currentMatchIdx, allMatches.length - 1) : 0;
  const currentMatch = allMatches[safeMatchIdx] ?? null;

  const displayPages = useMemo(() => {
    if (!searchRegex) return pagesInRange;
    return pagesInRange.filter((p) => countMatches(getEff(p), searchRegex) > 0);
  }, [pagesInRange, searchRegex, getEff]);

  const rangeText = useMemo(() => pagesInRange.map((p) => p.text).join("\n\n"), [pagesInRange]);
  const wordCount = useMemo(() => rangeText.trim() ? rangeText.trim().split(/\s+/).length : 0, [rangeText]);
  const lineCount = useMemo(() => rangeText ? rangeText.split("\n").length : 0, [rangeText]);
  const readingMinutes = Math.ceil(wordCount / WORDS_PER_MINUTE);
  const detectedLang = useMemo(() => detectLanguage(rangeText), [rangeText]);
  const downloadPages = useMemo(() => {
    const bm = pagesInRange.filter((p) => bookmarkedPages.has(p.page_number));
    const rest = pagesInRange.filter((p) => !bookmarkedPages.has(p.page_number));
    return [...bm, ...rest];
  }, [pagesInRange, bookmarkedPages]);

  // Actions
  function navigateToMatch(idx: number) {
    if (!allMatches.length) return;
    const c = (idx + allMatches.length) % allMatches.length;
    setCurrentMatchIdx(c);
    const match = allMatches[c];
    if (!match) return;
    setCollapsedPages((prev) => { const n = new Set(prev); n.delete(match.pageNum); return n; });
    setTimeout(() => document.getElementById(`page-${match.pageNum}`)?.scrollIntoView({ behavior: "smooth", block: "center" }), 50);
  }

  function replaceCurrent() {
    if (!currentMatch || !searchRegex || result?.kind !== "pdf") return;
    const { pageNum, localIdx } = currentMatch;
    const page = result.pages.find((p) => p.page_number === pageNum);
    if (!page) return;
    const text = getEff(page);
    let count = 0;
    const newText = text.replace(searchRegex, (match) => count++ === localIdx ? replaceWith : match);
    setModifiedTexts((prev) => new Map(prev).set(pageNum, newText));
  }

  function replaceAll() {
    if (!searchRegex || result?.kind !== "pdf") return;
    const updates = new Map(modifiedTexts);
    for (const page of pagesInRange) {
      const text = getEff(page);
      const newText = text.replace(searchRegex, replaceWith);
      if (newText !== text) updates.set(page.page_number, newText);
    }
    setModifiedTexts(updates); setQuery("");
  }

  function handlePrint() {
    const printWin = window.open("", "_blank");
    if (!printWin) return;
    const content = result?.kind === "pdf"
      ? pagesInRange.map((p) => `<h2>Page ${p.page_number}</h2><pre>${getEff(p)}</pre>`).join("<hr/>")
      : `<pre>${(result as ImageResult | null)?.text ?? ""}</pre>`;
    printWin.document.write(`<!DOCTYPE html><html><head><title>${result?.filename ?? "Print"}</title><style>body{font-family:Georgia,serif;max-width:800px;margin:0 auto;padding:2rem;line-height:1.7}h2{margin-top:2rem;border-bottom:1px solid #eee;padding-bottom:.5rem}pre{white-space:pre-wrap;font-family:inherit}hr{border:none;border-top:1px solid #eee;margin:2rem 0}@media print{@page{margin:2cm}}</style></head><body>${content}</body></html>`);
    printWin.document.close(); printWin.print();
  }

  function exportSummary() {
    if (result?.kind !== "pdf") return;
    const w = window.open("", "_blank");
    if (!w) return;
    w.document.write(buildSummaryHtml(result, wordCount, detectedLang));
    w.document.close(); w.print();
  }

  function copyAll() {
    const text = result?.kind === "pdf" ? buildTxt(downloadPages, getEff) : (result as ImageResult | null)?.text ?? "";
    navigator.clipboard.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 2000); });
  }

  function copyAsMarkdown() {
    if (result?.kind !== "pdf") return;
    navigator.clipboard.writeText(buildMd(result.filename, downloadPages, getEff));
  }

  async function processFile(file: File) {
    const ext = getExtension(file.name);
    if (!ACCEPTED_EXTS.includes(ext)) { setErrorMsg(`Unsupported file type "${ext}".`); setStatus("error"); return; }
    setStatus("loading"); setResult(null); setErrorMsg("");
    try { setResult(await fetchExtract(file)); setStatus("success"); }
    catch (e: unknown) { setErrorMsg(e instanceof Error ? e.message : "Error"); setStatus("error"); }
  }

  const onFileDrop = useCallback((e: React.DragEvent) => { e.preventDefault(); setFileDragging(false); const f = e.dataTransfer.files[0]; if (f) processFile(f); }, []);
  const onFileDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setFileDragging(true); }, []);
  const onFileDragLeave = useCallback(() => setFileDragging(false), []);
  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) { const f = e.target.files?.[0]; if (f) processFile(f); e.target.value = ""; }

  function reset() {
    setStatus("idle"); setResult(null); setErrorMsg(""); setQuery("");
    setPageOrder([]); setCollapsedPages(new Set()); setBookmarkedPages(new Set());
    setModifiedTexts(new Map()); setRedactTerms([]);
  }

  async function processMergeFiles(files: FileList) {
    setMergeStatus("loading"); setMergedSections([]); setMergeError("");
    const sections: MergeSection[] = [];
    try {
      for (const file of Array.from(files)) {
        if (!ACCEPTED_EXTS.includes(getExtension(file.name))) continue;
        const r = await fetchExtract(file);
        sections.push(r.kind === "pdf"
          ? { filename: r.filename, kind: "pdf", pages: r.pages }
          : { filename: r.filename, kind: "image", pages: [{ page_number: 1, text: r.text, method: "direct", character_count: r.text.length }] });
      }
      if (!sections.length) throw new Error("No valid files processed.");
      setMergedSections(sections); setMergeStatus("success");
    } catch (e: unknown) { setMergeError(e instanceof Error ? e.message : "Error"); setMergeStatus("error"); }
  }

  async function processSpreadsheet() {
    const validFields = ssFields.filter((f) => f.trim());
    if (!validFields.length) { setSsError("Add at least one field."); return; }
    if (!ssFiles.length) { setSsError("Upload at least one file."); return; }
    setSsStatus("loading"); setSsError(""); setSsPreview(null); setSsContext(null);
    try {
      const fd = new FormData();
      for (const file of ssFiles) fd.append("files", file);
      fd.append("fields", JSON.stringify(validFields));
      fd.append("method", ssMethod);
      const res = await fetch(`${API_BASE}/extract-fields`, { method: "POST", body: fd });
      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: "Extraction failed" }));
        throw new Error(data.detail ?? "Extraction failed");
      }
      const data = await res.json();
      setSsPreview(data.rows);
      setSsContext(data.context ?? null);
      setSsStatus("success");
    } catch (e: unknown) {
      setSsError(e instanceof Error ? e.message : "Error");
      setSsStatus("error");
    }
  }

  async function autoDetectFields() {
    if (!ssFiles.length) { setSsError("Upload at least one file first."); return; }
    setSsDetecting(true); setSsError("");
    try {
      const fd = new FormData();
      fd.append("file", ssFiles[0]);
      const res = await fetch(`${API_BASE}/auto-detect-fields`, { method: "POST", body: fd });
      if (!res.ok) throw new Error("Auto-detect failed");
      const data = await res.json();
      const fields = (data.detected_fields ?? []).map((d: { field: string }) => d.field);
      if (fields.length) setSsFields(fields);
      else setSsError("No fields detected. Try a different document.");
    } catch (e: unknown) {
      setSsError(e instanceof Error ? e.message : "Auto-detect failed");
    } finally {
      setSsDetecting(false);
    }
  }

  function downloadSpreadsheet() {
    if (!ssPreview || !ssPreview.length) return;
    const headers = Object.keys(ssPreview[0]);
    const csvEscape = (v: string) => `"${v.replace(/"/g, '""')}"`;
    const lines = [
      headers.map(csvEscape).join(","),
      ...ssPreview.map((row) => headers.map((h) => csvEscape(row[h] ?? "")).join(",")),
    ];
    triggerDownload(lines.join("\n"), "extracted_data.csv", "text/csv");
  }

  function handlePageDragStart(idx: number) { dragSrcIdxRef.current = idx; setDraggingPageNum(pageOrder[idx]); }
  function handlePageDragOver(e: React.DragEvent, idx: number) {
    e.preventDefault();
    const src = dragSrcIdxRef.current;
    if (src === null || src === idx) return;
    const newOrder = [...pageOrder];
    const [moved] = newOrder.splice(src, 1);
    newOrder.splice(idx, 0, moved);
    setPageOrder(newOrder); dragSrcIdxRef.current = idx;
  }
  function handlePageDragEnd() { dragSrcIdxRef.current = null; setDraggingPageNum(null); }
  function scrollToPage(n: number) {
    setCollapsedPages((prev) => { const next = new Set(prev); next.delete(n); return next; });
    setTimeout(() => document.getElementById(`page-${n}`)?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  }
  function expandAll() { setCollapsedPages(new Set()); }
  function collapseAll() { setCollapsedPages(new Set(displayPages.map((p) => p.page_number))); }

  const dark = darkMode;
  const btnCls = dark
    ? "bg-gray-700 border-gray-600 text-gray-200 hover:bg-gray-600"
    : "bg-white border-gray-300 text-gray-700 hover:bg-gray-50";
  const btnActiveCls = dark
    ? "bg-blue-900 border-blue-700 text-blue-300"
    : "bg-blue-50 border-blue-300 text-blue-700";

  // Download options
  const base = result?.filename?.replace(/\.[^.]+$/, "") ?? "doc";
  const downloadOpts = result?.kind === "pdf" ? [
    { label: "Plain Text (.txt)", ext: "txt", mime: "text/plain", build: () => buildTxt(downloadPages, getEff) },
    { label: "JSON (.json)", ext: "json", mime: "application/json", build: () => buildJson(result, downloadPages, getEff) },
    { label: "Markdown (.md)", ext: "md", mime: "text/markdown", build: () => buildMd(result.filename, downloadPages, getEff) },
  ] : [
    { label: "Plain Text (.txt)", ext: "txt", mime: "text/plain", build: () => (result as ImageResult).text },
  ];

  return (
    <DarkCtx.Provider value={darkMode}>
      <div className={`min-h-screen transition-colors ${dark ? "bg-gray-950 text-gray-100" : "bg-gray-50 text-gray-900"}`}>

        {/* ── Navbar ── */}
        <header className={`border-b sticky top-0 z-30 backdrop-blur ${dark ? "bg-gray-900/95 border-gray-800" : "bg-white/95 border-gray-200"}`}>
          <div className="max-w-5xl mx-auto px-6 h-14 flex items-center gap-4">
            <span className={`text-base font-bold tracking-tight ${dark ? "text-white" : "text-gray-900"}`}>doc-to-text</span>

            {/* Mode tabs */}
            <div className={`flex items-center rounded-lg border p-0.5 ml-3 ${dark ? "border-gray-700 bg-gray-800" : "border-gray-200 bg-gray-100"}`}>
              {(["single", "compare", "merge", "spreadsheet"] as AppMode[]).map((m) => (
                <button key={m} onClick={() => setAppMode(m)}
                  className={`px-3 py-1 rounded-md text-xs font-medium transition-all capitalize ${appMode === m
                    ? dark ? "bg-gray-600 text-white shadow-sm" : "bg-white text-gray-900 shadow-sm"
                    : dark ? "text-gray-500 hover:text-gray-300" : "text-gray-500 hover:text-gray-700"}`}>
                  {m}
                </button>
              ))}
            </div>

            <div className="ml-auto flex items-center gap-3">
              {status === "success" && appMode === "single" && (
                <button onClick={reset} className={`text-xs font-medium transition-colors ${dark ? "text-gray-500 hover:text-gray-300" : "text-gray-400 hover:text-gray-700"}`}>
                  &larr; New file
                </button>
              )}
              {/* Dark mode toggle */}
              <button onClick={() => setDarkMode((d) => !d)} title={dark ? "Light mode" : "Dark mode"}
                className={`p-2 rounded-lg transition-colors ${dark ? "text-gray-500 hover:text-yellow-400 hover:bg-gray-800" : "text-gray-400 hover:text-gray-700 hover:bg-gray-100"}`}>
                {dark
                  ? <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" /></svg>
                  : <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" /></svg>}
              </button>
            </div>
          </div>
        </header>

        <main className="max-w-5xl mx-auto px-6">

          {/* ══ COMPARE ══ */}
          {appMode === "compare" && (
            <div className="grid grid-cols-2 gap-5 py-8">
              <ComparePanel label="Document A" />
              <ComparePanel label="Document B" />
            </div>
          )}

          {/* ══ MERGE ══ */}
          {appMode === "merge" && (
            <div className="py-8 space-y-5">
              {mergeStatus !== "success" && (
                <div onClick={() => mergeInputRef.current?.click()}
                  className={`cursor-pointer rounded-2xl border-2 border-dashed flex flex-col items-center justify-center gap-4 py-20 text-center transition-all ${dark ? "border-gray-700 hover:border-gray-600 hover:bg-gray-900" : "border-gray-300 hover:border-blue-400 hover:bg-blue-50/30"}`}>
                  <input ref={mergeInputRef} type="file" multiple accept={ACCEPTED_EXTS.join(",")} className="hidden"
                    onChange={(e) => { if (e.target.files?.length) processMergeFiles(e.target.files); e.target.value = ""; }} />
                  <svg className={`w-12 h-12 ${dark ? "text-gray-700" : "text-blue-200"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}><path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" /></svg>
                  <div>
                    <p className={`text-base font-medium ${dark ? "text-gray-400" : "text-gray-600"}`}>Select multiple files to merge</p>
                    <p className={`text-sm mt-1 ${dark ? "text-gray-600" : "text-gray-400"}`}>PDF, PNG, JPG, JPEG, TIFF, BMP</p>
                  </div>
                  {mergeStatus === "error" && <p className="text-sm text-red-500">{mergeError}</p>}
                </div>
              )}
              {mergeStatus === "loading" && <div className="flex justify-center py-20"><div className="w-8 h-8 border-3 border-blue-500 border-t-transparent rounded-full animate-spin" /></div>}
              {mergeStatus === "success" && mergedSections.length > 0 && (
                <div className="space-y-5">
                  <div className="flex items-center gap-4">
                    <p className={`text-sm font-semibold ${dark ? "text-gray-300" : "text-gray-700"}`}>{mergedSections.length} files merged</p>
                    <button onClick={() => { setMergedSections([]); setMergeStatus("idle"); }} className={`text-xs font-medium ${dark ? "text-gray-600 hover:text-gray-400" : "text-gray-400 hover:text-gray-600"}`}>&larr; Start over</button>
                    <button onClick={() => triggerDownload(buildMergeTxt(mergedSections), "merged.txt", "text/plain")}
                      className="ml-auto inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 transition-colors">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" /></svg>
                      Download merged.txt
                    </button>
                  </div>
                  {mergedSections.map((section, si) => (
                    <div key={si} className={`rounded-2xl border overflow-hidden ${dark ? "bg-gray-900 border-gray-800" : "bg-white border-gray-200"}`}>
                      <div className={`px-5 py-3 border-b ${dark ? "bg-gray-800 border-gray-700" : "bg-blue-50 border-blue-100"}`}>
                        <p className={`text-sm font-semibold ${dark ? "text-gray-300" : "text-blue-700"}`}>{section.filename}</p>
                      </div>
                      <div className="p-5 space-y-4">
                        {section.pages.map((p) => (
                          <div key={p.page_number}>
                            <p className={`text-xs font-medium mb-2 ${dark ? "text-gray-600" : "text-gray-400"}`}>Page {p.page_number}</p>
                            <p className={`text-sm whitespace-pre-wrap leading-relaxed ${dark ? "text-gray-300" : "text-gray-700"}`}>{p.text || "—"}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ══ SPREADSHEET ══ */}
          {appMode === "spreadsheet" && (
            <div className="py-8 space-y-6">
              <h2 className={`text-lg font-bold ${dark ? "text-white" : "text-gray-900"}`}>Structured Spreadsheet Extraction</h2>
              <p className={`text-sm ${dark ? "text-gray-500" : "text-gray-500"}`}>
                Upload PDFs, define the fields you want to extract, and download a spreadsheet with the results.
              </p>

              {/* Field builder */}
              <div className={`rounded-2xl border p-5 space-y-4 ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
                <h3 className={`text-sm font-semibold ${dark ? "text-gray-300" : "text-gray-700"}`}>Fields to Extract</h3>
                <div className="space-y-2">
                  {ssFields.map((field, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <input
                        type="text"
                        value={field}
                        onChange={(e) => { const f = [...ssFields]; f[i] = e.target.value; setSsFields(f); }}
                        placeholder={`Field ${i + 1} (e.g. Patient Name, DOB, Claim Number…)`}
                        className={`flex-1 rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${dark ? "bg-gray-700 border-gray-600 text-gray-100 placeholder-gray-500" : "bg-white border-gray-300 text-gray-900 placeholder-gray-400"}`}
                      />
                      {ssFields.length > 1 && (
                        <button onClick={() => setSsFields(ssFields.filter((_, j) => j !== i))}
                          className={`p-2 rounded-lg transition-colors ${dark ? "text-gray-600 hover:text-red-400 hover:bg-gray-700" : "text-gray-400 hover:text-red-500 hover:bg-gray-100"}`}>
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                        </button>
                      )}
                    </div>
                  ))}
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  <button onClick={() => setSsFields([...ssFields, ""])}
                    className={`inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${btnCls}`}>
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" /></svg>
                    Add Field
                  </button>
                  <button onClick={autoDetectFields} disabled={ssDetecting || !ssFiles.length}
                    className={`inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors disabled:opacity-40 ${btnCls}`}>
                    {ssDetecting ? (
                      <><div className="w-3.5 h-3.5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" /> Detecting…</>
                    ) : (
                      <><svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" /></svg>
                      Auto-Detect Fields</>
                    )}
                  </button>
                </div>
                <p className={`text-xs ${dark ? "text-gray-600" : "text-gray-400"}`}>Upload a file first, then click Auto-Detect to find extractable fields automatically.</p>
              </div>

              {/* File upload */}
              <div className={`rounded-2xl border p-5 space-y-4 ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
                <h3 className={`text-sm font-semibold ${dark ? "text-gray-300" : "text-gray-700"}`}>Upload Documents</h3>
                <div
                  onClick={() => ssInputRef.current?.click()}
                  className={`cursor-pointer rounded-xl border-2 border-dashed flex flex-col items-center justify-center gap-3 py-12 text-center transition-all ${dark ? "border-gray-700 hover:border-gray-600 hover:bg-gray-900" : "border-gray-300 hover:border-blue-400 hover:bg-blue-50/30"}`}>
                  <input ref={ssInputRef} type="file" multiple accept={ACCEPTED_EXTS.join(",")} className="hidden"
                    onChange={(e) => { if (e.target.files?.length) setSsFiles(Array.from(e.target.files)); e.target.value = ""; }} />
                  <svg className={`w-10 h-10 ${dark ? "text-gray-700" : "text-blue-200"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                  </svg>
                  <p className={`text-sm ${dark ? "text-gray-500" : "text-gray-500"}`}>Click to select PDFs or images (multiple allowed)</p>
                </div>
                {ssFiles.length > 0 && (
                  <div className="space-y-1.5">
                    {ssFiles.map((f, i) => (
                      <div key={i} className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${dark ? "bg-gray-700/50 text-gray-300" : "bg-gray-50 text-gray-700"}`}>
                        <svg className="w-4 h-4 flex-shrink-0 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" /></svg>
                        <span className="truncate flex-1">{f.name}</span>
                        <span className={`text-xs ${dark ? "text-gray-600" : "text-gray-400"}`}>{(f.size / 1024).toFixed(0)} KB</span>
                        <button onClick={() => setSsFiles(ssFiles.filter((_, j) => j !== i))}
                          className={`p-1 rounded transition-colors ${dark ? "text-gray-600 hover:text-red-400" : "text-gray-400 hover:text-red-500"}`}>
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Method selector + Extract button */}
              <div className="flex items-center gap-3 flex-wrap">
                <div className={`flex items-center rounded-lg border p-0.5 ${dark ? "border-gray-700 bg-gray-800" : "border-gray-200 bg-gray-100"}`}>
                  {(["auto", "block"] as const).map((m) => (
                    <button key={m} onClick={() => setSsMethod(m)}
                      className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${ssMethod === m
                        ? dark ? "bg-gray-600 text-white shadow-sm" : "bg-white text-gray-900 shadow-sm"
                        : dark ? "text-gray-500 hover:text-gray-300" : "text-gray-500 hover:text-gray-700"}`}>
                      {m === "auto" ? "Smart Extract" : "Full Block"}
                    </button>
                  ))}
                </div>
                <button onClick={processSpreadsheet} disabled={ssStatus === "loading"}
                  className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700 transition-colors disabled:opacity-50">
                  {ssStatus === "loading" ? (
                    <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> Extracting…</>
                  ) : (
                    <><svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125m-9.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-7.5A1.125 1.125 0 0112 18.375m9.75-12.75c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125m19.5 0v1.5c0 .621-.504 1.125-1.125 1.125M2.25 5.625v1.5c0 .621.504 1.125 1.125 1.125m0 0h17.25m-17.25 0h7.5c.621 0 1.125.504 1.125 1.125M3.375 8.25c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125m17.25-3.75h-7.5c-.621 0-1.125.504-1.125 1.125m8.625-1.125c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h7.5m-7.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125M12 10.875v-1.5m0 1.5c0 .621-.504 1.125-1.125 1.125M12 10.875c0 .621.504 1.125 1.125 1.125m-2.25 0c.621 0 1.125.504 1.125 1.125M12 12h7.5m-7.5 0c-.621 0-1.125.504-1.125 1.125M21.375 12c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125M12 13.125v1.5m0-1.5c0-.621-.504-1.125-1.125-1.125M12 13.125c0-.621.504-1.125 1.125-1.125m-2.25 0c-.621 0-1.125.504-1.125 1.125m0 1.5v-1.5m0 0c0-.621.504-1.125 1.125-1.125m-1.125 2.625c0 .621.504 1.125 1.125 1.125M3.375 15h7.5m-7.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125" /></svg>
                    Extract Fields</>
                  )}
                </button>
                {ssStatus === "success" && (
                  <button onClick={() => { setSsStatus("idle"); setSsPreview(null); setSsContext(null); setSsFiles([]); setSsFields([""]); }}
                    className={`text-xs font-medium ${dark ? "text-gray-600 hover:text-gray-400" : "text-gray-400 hover:text-gray-600"}`}>&larr; Start over</button>
                )}
              </div>

              {ssError && <p className="text-sm text-red-500">{ssError}</p>}

              {/* Preview table */}
              {ssPreview && ssPreview.length > 0 && (() => {
                const headers = Object.keys(ssPreview[0]);
                return (
                  <div className={`rounded-2xl border overflow-hidden ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
                    <div className={`flex items-center justify-between px-5 py-3 border-b ${dark ? "border-gray-700" : "border-gray-100"}`}>
                      <div className="flex items-center gap-3">
                        <h3 className={`text-sm font-semibold ${dark ? "text-gray-300" : "text-gray-700"}`}>Extracted Data Preview</h3>
                        {ssContext && (
                          <button onClick={() => setSsShowContext((v) => !v)}
                            className={`text-xs px-2.5 py-1 rounded-lg border font-medium transition-colors ${ssShowContext ? btnActiveCls : btnCls}`}>
                            {ssShowContext ? "Hide Context" : "Show Context"}
                          </button>
                        )}
                      </div>
                      <button onClick={downloadSpreadsheet}
                        className="inline-flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700 transition-colors">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" /></svg>
                        Download CSV
                      </button>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className={dark ? "bg-gray-700/50" : "bg-gray-50"}>
                            {headers.map((h) => (
                              <th key={h} className={`px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider ${dark ? "text-gray-400" : "text-gray-500"}`}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {ssPreview.map((row, ri) => (
                            <tr key={ri} className={`border-t ${dark ? "border-gray-700" : "border-gray-100"} ${ri % 2 === 0 ? "" : dark ? "bg-gray-700/20" : "bg-gray-50/50"}`}>
                              {headers.map((h) => {
                                const val = row[h];
                                const ctx = ssContext?.[ri]?.[h];
                                return (
                                  <td key={h} className={`px-4 py-3 ${dark ? "text-gray-300" : "text-gray-700"} ${val ? "" : dark ? "text-gray-700 italic" : "text-gray-300 italic"}`}>
                                    <div>{val || "Not found"}</div>
                                    {ssShowContext && ctx && (
                                      <div className={`mt-1.5 text-xs px-2 py-1.5 rounded-lg font-mono leading-relaxed ${dark ? "bg-gray-900/60 text-gray-500" : "bg-yellow-50 text-gray-500 border border-yellow-100"}`}>
                                        {ctx}
                                      </div>
                                    )}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          {/* ══ SINGLE ══ */}
          {appMode === "single" && (
            <>
              {/* Drop zone */}
              {(status === "idle" || status === "error") && (
                <div className="py-20">
                  <div onDrop={onFileDrop} onDragOver={onFileDragOver} onDragLeave={onFileDragLeave}
                    onClick={() => inputRef.current?.click()}
                    className={`cursor-pointer rounded-2xl border-2 border-dashed flex flex-col items-center justify-center gap-5 py-24 text-center transition-all ${fileDragging ? "border-blue-400 bg-blue-50 dark:bg-blue-900/10" : dark ? "border-gray-700 hover:border-gray-600 hover:bg-gray-900" : "border-gray-300 hover:border-blue-400 hover:bg-blue-50/30"}`}>
                    <input ref={inputRef} type="file" accept={ACCEPTED_EXTS.join(",")} className="hidden" onChange={onFileChange} />
                    <svg className={`w-14 h-14 ${fileDragging ? "text-blue-400" : dark ? "text-gray-700" : "text-blue-200"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                    </svg>
                    <div>
                      <p className={`text-lg font-medium ${dark ? "text-gray-400" : "text-gray-600"}`}>{fileDragging ? "Drop file here" : "Drop a file or click to browse"}</p>
                      <p className={`text-sm mt-2 ${dark ? "text-gray-600" : "text-gray-400"}`}>PDF, PNG, JPG, JPEG, TIFF, BMP</p>
                    </div>
                  </div>
                  {status === "error" && (
                    <div className="mt-4 text-center">
                      <p className="text-sm text-red-500">{errorMsg}</p>
                    </div>
                  )}
                </div>
              )}

              {/* Loading */}
              {status === "loading" && (
                <div className="flex flex-col items-center justify-center py-40 gap-4">
                  <div className="w-10 h-10 border-3 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  <p className={`text-sm ${dark ? "text-gray-500" : "text-gray-400"}`}>Extracting text…</p>
                </div>
              )}

              {/* ══ SUCCESS ══ */}
              {status === "success" && result && (
                <div className="py-8 space-y-6">

                  {/* ── Filename + Info icon ── */}
                  <div className="flex items-center gap-3 flex-wrap">
                    <h2 className={`text-xl font-bold ${dark ? "text-white" : "text-gray-900"}`}>{result.filename}</h2>
                    {result.kind === "pdf" && (
                      <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${dark ? "bg-blue-900/40 text-blue-300" : "bg-blue-100 text-blue-700"}`}>
                        {detectedLang}
                      </span>
                    )}

                    {/* Info icon + dropdown */}
                    {result.kind === "pdf" && (
                      <div className="relative" ref={infoRef}>
                        <button onClick={() => setShowInfo((v) => !v)} title="Document details"
                          className={`p-1.5 rounded-lg transition-colors ${showInfo ? dark ? "bg-blue-900/40 text-blue-400" : "bg-blue-100 text-blue-600" : dark ? "text-gray-600 hover:text-gray-300 hover:bg-gray-800" : "text-gray-400 hover:text-gray-600 hover:bg-gray-100"}`}>
                          <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
                          </svg>
                        </button>
                        {showInfo && (
                          <div className={`absolute left-0 top-full mt-2 w-72 rounded-xl border shadow-xl z-50 p-5 ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
                            <p className={`text-xs font-semibold uppercase tracking-wider mb-4 ${dark ? "text-gray-500" : "text-gray-400"}`}>Document Details</p>
                            <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                              {([
                                ["Pages", result.total_pages.toString()],
                                ["Words", fmt(wordCount)],
                                ["Characters", fmt(result.total_characters)],
                                ["Lines", fmt(lineCount)],
                                ["Reading time", readingMinutes < 1 ? "< 1 min" : `~${readingMinutes} min`],
                                ["Extract time", `${result.extraction_time_seconds}s`],
                                ["Language", detectedLang],
                                ["OCR pages", result.pages.filter((p) => p.method === "ocr").length.toString()],
                              ] as [string, string][]).map(([label, value]) => (
                                <div key={label}>
                                  <p className={`text-xs ${dark ? "text-gray-500" : "text-gray-400"}`}>{label}</p>
                                  <p className={`text-sm font-semibold mt-0.5 ${dark ? "text-white" : "text-gray-900"}`}>{value}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {result.kind === "pdf" && bookmarkedPages.size > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap ml-auto">
                        <span className={`text-xs mr-1 ${dark ? "text-gray-600" : "text-gray-400"}`}>Bookmarks:</span>
                        {pagesInRange.filter((p) => bookmarkedPages.has(p.page_number)).map((p) => (
                          <button key={p.page_number} onClick={() => scrollToPage(p.page_number)}
                            className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium transition-colors ${dark ? "bg-amber-900/40 text-amber-400 hover:bg-amber-900/60" : "bg-amber-100 text-amber-700 hover:bg-amber-200"}`}>
                            <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 24 24"><path d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0111.186 0z" /></svg>
                            {p.page_number}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* ── Search bar ── */}
                  {result.kind === "pdf" && (
                    <div className={`rounded-2xl border overflow-hidden ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
                      <div className="flex items-center gap-2 px-4 py-3">
                        <svg className={`w-5 h-5 flex-shrink-0 ${dark ? "text-gray-600" : "text-blue-400"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 15.803a7.5 7.5 0 0010.607 0z" />
                        </svg>
                        <input ref={searchInputRef} type="text" value={query} onChange={(e) => setQuery(e.target.value)}
                          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); navigateToMatch(e.shiftKey ? safeMatchIdx - 1 : safeMatchIdx + 1); } if (e.key === "Escape") setQuery(""); }}
                          placeholder="Search extracted text… (Enter = next, Shift+Enter = prev)"
                          className={`flex-1 bg-transparent text-sm focus:outline-none ${dark ? "text-gray-200 placeholder-gray-600" : "text-gray-800 placeholder-gray-400"}`} />

                        {/* Match nav */}
                        {query.trim() && (
                          <div className="flex items-center gap-1">
                            <span className={`text-xs tabular-nums mr-1 ${dark ? "text-gray-500" : "text-gray-400"}`}>
                              {allMatches.length === 0 ? "0 results" : `${safeMatchIdx + 1} of ${allMatches.length}`}
                            </span>
                            <button onClick={() => navigateToMatch(safeMatchIdx - 1)} disabled={!allMatches.length}
                              className={`p-1 rounded-md transition-colors disabled:opacity-30 ${dark ? "text-gray-500 hover:text-white hover:bg-gray-700" : "text-gray-400 hover:text-gray-700 hover:bg-gray-100"}`}>
                              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}><path strokeLinecap="round" strokeLinejoin="round" d="M4.5 15.75l7.5-7.5 7.5 7.5" /></svg>
                            </button>
                            <button onClick={() => navigateToMatch(safeMatchIdx + 1)} disabled={!allMatches.length}
                              className={`p-1 rounded-md transition-colors disabled:opacity-30 ${dark ? "text-gray-500 hover:text-white hover:bg-gray-700" : "text-gray-400 hover:text-gray-700 hover:bg-gray-100"}`}>
                              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" /></svg>
                            </button>
                          </div>
                        )}

                        {query && (
                          <button onClick={() => setQuery("")} className={`p-1 rounded-md ${dark ? "text-gray-600 hover:text-gray-300" : "text-gray-300 hover:text-gray-500"}`}>
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                          </button>
                        )}

                        {/* Toggles */}
                        <div className={`w-px h-5 mx-1 ${dark ? "bg-gray-700" : "bg-gray-200"}`} />
                        <button onClick={() => setCaseSensitive((v) => !v)} title="Match case"
                          className={`px-2 py-1 rounded-md text-xs font-bold transition-colors ${caseSensitive ? btnActiveCls : btnCls} border`}>Aa</button>
                        <button onClick={() => setWholeWord((v) => !v)} title="Whole word"
                          className={`px-2 py-1 rounded-md text-xs font-bold transition-colors ${wholeWord ? btnActiveCls : btnCls} border`}>W</button>
                        <button onClick={() => setShowReplace((v) => !v)} title="Find & Replace"
                          className={`px-2 py-1 rounded-md text-xs font-bold transition-colors ${showReplace ? btnActiveCls : btnCls} border`}>&#8597;</button>
                      </div>

                      {/* Replace row */}
                      {showReplace && (
                        <div className={`flex items-center gap-2 px-4 py-3 border-t ${dark ? "border-gray-700" : "border-gray-100"}`}>
                          <svg className={`w-5 h-5 flex-shrink-0 ${dark ? "text-gray-600" : "text-gray-400"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" /></svg>
                          <input type="text" value={replaceWith} onChange={(e) => setReplaceWith(e.target.value)}
                            placeholder="Replace with…"
                            className={`flex-1 bg-transparent text-sm focus:outline-none ${dark ? "text-gray-200 placeholder-gray-600" : "text-gray-800 placeholder-gray-400"}`} />
                          <button onClick={replaceCurrent} disabled={!currentMatch}
                            className={`text-xs px-3 py-1.5 rounded-lg border font-medium transition-colors disabled:opacity-40 ${btnCls}`}>Replace</button>
                          <button onClick={replaceAll} disabled={!allMatches.length}
                            className="text-xs px-3 py-1.5 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 transition-colors disabled:opacity-40">Replace All</button>
                          {modifiedTexts.size > 0 && (
                            <>
                              <span className={`text-xs ${dark ? "text-gray-600" : "text-gray-400"}`}>{modifiedTexts.size} modified</span>
                              <button onClick={() => setModifiedTexts(new Map())} className="text-xs text-red-400 hover:text-red-500 font-medium">Reset</button>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* ── Actions bar ── */}
                  <div className="flex items-center gap-2 flex-wrap">
                    {/* Copy All */}
                    <button onClick={copyAll}
                      className={`inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${btnCls}`}>
                      {copied
                        ? <svg className="w-4 h-4 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                        : <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184" /></svg>}
                      {copied ? "Copied!" : "Copy All"}
                    </button>

                    {/* Download dropdown */}
                    <div className="relative" ref={downloadRef}>
                      <button onClick={() => setShowDownload((v) => !v)}
                        className={`inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${showDownload ? btnActiveCls : btnCls}`}>
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" /></svg>
                        Download
                        <svg className="w-3 h-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" /></svg>
                      </button>
                      {showDownload && (
                        <div className={`absolute left-0 top-full mt-2 w-56 rounded-xl border shadow-xl z-40 overflow-hidden ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
                          <div className="p-1.5">
                            {downloadOpts.map((o) => (
                              <button key={o.ext}
                                onClick={() => { triggerDownload(o.build(), `${base}.${o.ext}`, o.mime); setShowDownload(false); }}
                                className={`flex w-full items-center gap-3 px-3 py-2.5 text-sm rounded-lg transition-colors ${dark ? "text-gray-300 hover:bg-gray-700" : "text-gray-700 hover:bg-gray-50"}`}>
                                <span className={`w-10 text-center rounded-md font-mono text-xs font-bold py-1 ${dark ? "bg-gray-700 text-gray-400" : "bg-blue-50 text-blue-600"}`}>.{o.ext}</span>
                                {o.label}
                              </button>
                            ))}
                            {result.kind === "pdf" && (
                              <>
                                <div className={`my-1.5 border-t ${dark ? "border-gray-700" : "border-gray-100"}`} />
                                <button onClick={() => { copyAsMarkdown(); setShowDownload(false); }}
                                  className={`flex w-full items-center gap-3 px-3 py-2.5 text-sm rounded-lg transition-colors ${dark ? "text-gray-300 hover:bg-gray-700" : "text-gray-700 hover:bg-gray-50"}`}>
                                  <span className={`w-10 text-center rounded-md font-mono text-xs font-bold py-1 ${dark ? "bg-gray-700 text-gray-400" : "bg-blue-50 text-blue-600"}`}>MD</span>
                                  Copy as Markdown
                                </button>
                              </>
                            )}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Print */}
                    <button onClick={handlePrint}
                      className={`inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${btnCls}`}>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M6.72 13.829c-.24.03-.48.062-.72.096m.72-.096a42.415 42.415 0 0110.56 0m-10.56 0L6.34 18m10.94-4.171c.24.03.48.062.72.096m-.72-.096L17.66 18m0 0l.229 2.523a1.125 1.125 0 01-1.12 1.227H7.231c-.662 0-1.18-.568-1.12-1.227L6.34 18m11.318 0h1.091A2.25 2.25 0 0021 15.75V9.456c0-1.081-.768-2.015-1.837-2.175a48.055 48.055 0 00-1.913-.247M6.34 18H5.25A2.25 2.25 0 013 15.75V9.456c0-1.081.768-2.015 1.837-2.175a48.041 48.041 0 011.913-.247m10.5 0a48.536 48.536 0 00-10.5 0m10.5 0V3.375c0-.621-.504-1.125-1.125-1.125h-8.25c-.621 0-1.125.504-1.125 1.125v3.659M18 10.5h.008v.008H18V10.5zm-3 0h.008v.008H15V10.5z" /></svg>
                      Print
                    </button>

                    {/* Export Summary */}
                    {result.kind === "pdf" && (
                      <button onClick={exportSummary}
                        className={`inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${btnCls}`}>
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25z" /></svg>
                        Export Summary
                      </button>
                    )}

                    {/* Spacer */}
                    <div className="flex-1" />

                    {/* View controls */}
                    {result.kind === "pdf" && (
                      <>
                        <button onClick={expandAll} className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${btnCls}`}>
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" /></svg>
                          Expand All
                        </button>
                        <button onClick={collapseAll} className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${btnCls}`}>
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 9V4.5M9 9H4.5M9 9L3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5l5.25 5.25" /></svg>
                          Collapse All
                        </button>
                        <button onClick={() => setShowPageNav((v) => !v)}
                          className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${showPageNav ? btnActiveCls : btnCls}`}>
                          Pages
                        </button>
                        <button onClick={() => setShowStats((v) => !v)}
                          className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${showStats ? btnActiveCls : btnCls}`}>
                          Stats
                        </button>
                        <button onClick={() => setShowRedact((v) => !v)}
                          className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${showRedact ? btnActiveCls : btnCls}`}>
                          Redact
                        </button>
                      </>
                    )}
                  </div>

                  {/* ── Page range ── */}
                  {result.kind === "pdf" && (
                    <div className={`flex items-center gap-3 text-xs ${dark ? "text-gray-500" : "text-gray-400"}`}>
                      <span>Page range:</span>
                      <input type="number" min={1} max={toPage} value={fromPage}
                        onChange={(e) => setFromPage(Math.max(1, Math.min(+e.target.value, toPage)))}
                        className={`w-14 rounded-lg border px-2 py-1 text-center text-xs font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 ${dark ? "bg-gray-800 border-gray-700 text-gray-300" : "bg-white border-gray-300 text-gray-700"}`} />
                      <span>to</span>
                      <input type="number" min={fromPage} max={result.total_pages} value={toPage}
                        onChange={(e) => setToPage(Math.max(fromPage, Math.min(+e.target.value, result.total_pages)))}
                        className={`w-14 rounded-lg border px-2 py-1 text-center text-xs font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 ${dark ? "bg-gray-800 border-gray-700 text-gray-300" : "bg-white border-gray-300 text-gray-700"}`} />
                      <span>of {result.total_pages}</span>
                    </div>
                  )}

                  {/* ── Page navigator (collapsible) ── */}
                  {showPageNav && result.kind === "pdf" && (() => {
                    const visiblePages = pageNavExpanded ? orderedPages : orderedPages.slice(0, PAGE_NAV_LIMIT);
                    const hasMore = orderedPages.length > PAGE_NAV_LIMIT;
                    return (
                      <div className={`rounded-2xl border p-4 ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
                        <div className="flex flex-wrap gap-1.5">
                          {visiblePages.map((p) => {
                            const inRange = p.page_number >= fromPage && p.page_number <= toPage;
                            const isBookmarked = bookmarkedPages.has(p.page_number);
                            return (
                              <button key={p.page_number} onClick={() => scrollToPage(p.page_number)}
                                className={`relative rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                                  !inRange
                                    ? dark ? "text-gray-800 cursor-default" : "text-gray-300 cursor-default"
                                    : isBookmarked
                                      ? dark ? "bg-amber-900/40 text-amber-400 hover:bg-amber-900/60" : "bg-amber-100 text-amber-700 hover:bg-amber-200"
                                      : dark ? "text-gray-400 hover:text-white hover:bg-gray-700" : "text-gray-600 hover:text-gray-900 hover:bg-blue-50"
                                }`}>
                                {p.page_number}
                              </button>
                            );
                          })}
                        </div>
                        {hasMore && (
                          <button onClick={() => setPageNavExpanded((v) => !v)}
                            className={`mt-3 text-xs font-medium transition-colors ${dark ? "text-blue-400 hover:text-blue-300" : "text-blue-600 hover:text-blue-700"}`}>
                            {pageNavExpanded ? "Show less" : `Show all ${orderedPages.length} pages`}
                          </button>
                        )}
                      </div>
                    );
                  })()}

                  {/* ── Stats / Redact panels ── */}
                  {(showStats || showRedact) && result.kind === "pdf" && (
                    <div className="space-y-4">
                      {showStats && <StatsPanel pages={pagesInRange} text={rangeText} />}
                      {showRedact && (
                        <RedactPanel terms={redactTerms}
                          onAdd={(t) => setRedactTerms((prev) => [...prev, t])}
                          onRemove={(i) => setRedactTerms((prev) => prev.filter((_, j) => j !== i))} />
                      )}
                    </div>
                  )}

                  {/* ── Document content ── */}
                  {result.kind === "pdf" ? (
                    <div className="space-y-4">
                      {displayPages.map((page, idx) => {
                        const effectiveText = getEff(page);
                        const isCollapsed = collapsedPages.has(page.page_number);
                        const isBookmarked = bookmarkedPages.has(page.page_number);
                        const isDragging = draggingPageNum === page.page_number;
                        const hasTable = detectTable(page.text);
                        const matchLocalIdx = currentMatch?.pageNum === page.page_number ? currentMatch.localIdx : -1;

                        return (
                          <div key={page.page_number} id={`page-${page.page_number}`}
                            draggable
                            onDragStart={() => handlePageDragStart(idx)}
                            onDragOver={(e) => handlePageDragOver(e, idx)}
                            onDragEnd={handlePageDragEnd}
                            className={`rounded-2xl border overflow-hidden scroll-mt-20 transition-all ${isDragging ? "opacity-30 scale-[0.98]" : ""} ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>

                            {/* Page header */}
                            <div className={`flex items-center gap-3 px-5 py-3 border-b cursor-pointer select-none ${dark ? "border-gray-700 hover:bg-gray-700/50" : "border-gray-100 hover:bg-gray-50"}`}
                              onClick={() => setCollapsedPages((prev) => { const n = new Set(prev); if (n.has(page.page_number)) n.delete(page.page_number); else n.add(page.page_number); return n; })}>

                              {/* Collapse arrow */}
                              <svg className={`w-4 h-4 transition-transform ${isCollapsed ? "-rotate-90" : ""} ${dark ? "text-gray-600" : "text-gray-400"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
                              </svg>

                              <span className={`text-sm font-semibold ${dark ? "text-gray-300" : "text-gray-700"}`}>Page {page.page_number}</span>

                              {page.method === "ocr" && (
                                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${dark ? "bg-amber-900/40 text-amber-400" : "bg-amber-100 text-amber-700"}`}>OCR</span>
                              )}
                              {hasTable && (
                                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${dark ? "bg-purple-900/40 text-purple-400" : "bg-purple-100 text-purple-700"}`}>Table</span>
                              )}

                              <span className={`text-xs ml-auto ${dark ? "text-gray-600" : "text-gray-400"}`}>{fmt(page.character_count)} chars</span>

                              {/* Bookmark */}
                              <button onClick={(e) => { e.stopPropagation(); setBookmarkedPages((prev) => { const n = new Set(prev); if (n.has(page.page_number)) n.delete(page.page_number); else n.add(page.page_number); return n; }); }}
                                className={`p-1 rounded transition-colors ${isBookmarked ? "text-amber-500" : dark ? "text-gray-700 hover:text-amber-400" : "text-gray-300 hover:text-amber-500"}`}>
                                <svg className="w-4 h-4" fill={isBookmarked ? "currentColor" : "none"} viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0111.186 0z" />
                                </svg>
                              </button>

                              {/* Copy page */}
                              <button onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(effectiveText); }}
                                className={`p-1 rounded transition-colors ${dark ? "text-gray-700 hover:text-gray-400" : "text-gray-300 hover:text-gray-500"}`}>
                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184" /></svg>
                              </button>

                              {/* Drag handle */}
                              <svg className={`w-4 h-4 cursor-grab ${dark ? "text-gray-700" : "text-gray-300"}`} fill="currentColor" viewBox="0 0 20 20">
                                <path d="M7 2a2 2 0 1 0 .001 4.001A2 2 0 0 0 7 2zm0 6a2 2 0 1 0 .001 4.001A2 2 0 0 0 7 8zm0 6a2 2 0 1 0 .001 4.001A2 2 0 0 0 7 14zm6-8a2 2 0 1 0-.001-4.001A2 2 0 0 0 13 6zm0 2a2 2 0 1 0 .001 4.001A2 2 0 0 0 13 8zm0 6a2 2 0 1 0 .001 4.001A2 2 0 0 0 13 14z" />
                              </svg>
                            </div>

                            {/* Page content */}
                            {!isCollapsed && (
                              <div className="px-5 py-4">
                                {effectiveText.trim() ? (
                                  <p className={`text-[15px] leading-[1.8] whitespace-pre-wrap font-sans ${dark ? "text-gray-200" : "text-gray-800"}`}>
                                    {highlightText(effectiveText, searchRegex, matchLocalIdx)}
                                  </p>
                                ) : (
                                  <p className={`text-sm italic py-4 ${dark ? "text-gray-700" : "text-gray-400"}`}>No text found on this page.</p>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                      {displayPages.length === 0 && searchRegex && (
                        <p className={`text-sm py-16 text-center ${dark ? "text-gray-700" : "text-gray-400"}`}>No pages match your search.</p>
                      )}
                    </div>
                  ) : (
                    /* Image result */
                    <div className={`rounded-2xl border p-6 ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
                      <p className={`text-[15px] leading-[1.8] whitespace-pre-wrap font-sans ${dark ? "text-gray-200" : "text-gray-800"}`}>
                        {(result as ImageResult).text || <span className={`italic ${dark ? "text-gray-700" : "text-gray-400"}`}>No text found in this image.</span>}
                      </p>
                    </div>
                  )}

                  <div className="h-20" />
                </div>
              )}
            </>
          )}
        </main>

        {/* Go to top */}
        {showGoTop && (
          <button onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            className={`fixed bottom-6 right-6 z-30 rounded-full p-3 shadow-lg transition-all hover:scale-110 ${dark ? "bg-gray-800 text-gray-400 hover:text-white border border-gray-700" : "bg-white text-blue-600 hover:text-blue-700 border border-gray-200 shadow-md"}`}
            title="Back to top">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4.5 15.75l7.5-7.5 7.5 7.5" /></svg>
          </button>
        )}
      </div>
    </DarkCtx.Provider>
  );
}
