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
type AppMode = "single" | "compare" | "merge";
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
        ? "bg-orange-300 text-orange-900 rounded-sm px-0.5"
        : "bg-yellow-200 text-yellow-900 rounded-sm px-0.5"}>
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
async function fetchExtract(file: File): Promise<ExtractResult> {
  const ext = getExtension(file.name);
  const endpoint = PDF_EXTS.includes(ext)
    ? "http://localhost:8000/extract-text"
    : "http://localhost:8000/extract-text-image";
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(endpoint, { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail ?? "Extraction failed");
  return PDF_EXTS.includes(ext) ? { kind: "pdf", ...data } : { kind: "image", ...data };
}

// ─── Small atoms ─────────────────────────────────────────────────────────────

function Sep() {
  const dark = useContext(DarkCtx);
  return <div className={`w-px h-4 mx-0.5 flex-shrink-0 ${dark ? "bg-gray-700" : "bg-gray-200"}`} />;
}

// Compact toolbar button — icon + optional label
function TBtn({
  onClick, active = false, title, children, disabled = false,
}: {
  onClick?: () => void; active?: boolean; title?: string;
  children: React.ReactNode; disabled?: boolean;
}) {
  const dark = useContext(DarkCtx);
  return (
    <button onClick={onClick} disabled={disabled} title={title}
      className={`inline-flex items-center gap-1.5 px-2 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap disabled:opacity-40 ${
        active
          ? dark ? "bg-blue-900/40 text-blue-300" : "bg-blue-50 text-blue-700"
          : dark ? "text-gray-400 hover:text-gray-100 hover:bg-gray-700" : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"
      }`}>
      {children}
    </button>
  );
}

// ─── DetailsPanel ─────────────────────────────────────────────────────────────

function DetailsPanel({
  result, wordCount, lineCount, readingMinutes, detectedLang, onClose,
}: {
  result: PdfResult; wordCount: number; lineCount: number;
  readingMinutes: number; detectedLang: string; onClose: () => void;
}) {
  const dark = useContext(DarkCtx);
  const ref = useRef<HTMLDivElement>(null);
  const ocrPages = result.pages.filter((p) => p.method === "ocr").length;

  useEffect(() => {
    function h(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    }
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, [onClose]);

  return (
    <div ref={ref}
      className={`absolute top-full right-0 mt-2 w-72 rounded-xl border shadow-xl z-50 p-5 ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
      <div className="grid grid-cols-2 gap-x-6 gap-y-4">
        {([
          ["Pages", result.total_pages.toString()],
          ["Words", fmt(wordCount)],
          ["Lines", fmt(lineCount)],
          ["Characters", fmt(result.total_characters)],
          ["Reading time", readingMinutes < 1 ? "< 1 min" : `~${readingMinutes} min`],
          ["Extract time", `${result.extraction_time_seconds}s`],
          ["Language", detectedLang],
          ["OCR pages", ocrPages.toString()],
        ] as [string, string][]).map(([label, value]) => (
          <div key={label}>
            <p className={`text-xs ${dark ? "text-gray-500" : "text-gray-400"}`}>{label}</p>
            <p className={`text-sm font-semibold mt-0.5 ${dark ? "text-white" : "text-gray-900"}`}>{value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── ViewOptionsMenu ──────────────────────────────────────────────────────────

function ViewOptionsMenu({
  showPageNav, onTogglePageNav,
  onExpandAll, onCollapseAll,
  fromPage, toPage, totalPages,
  onFromChange, onToChange,
  showStats, onToggleStats,
  showRedact, onToggleRedact,
}: {
  showPageNav: boolean; onTogglePageNav: () => void;
  onExpandAll: () => void; onCollapseAll: () => void;
  fromPage: number; toPage: number; totalPages: number;
  onFromChange: (v: number) => void; onToChange: (v: number) => void;
  showStats: boolean; onToggleStats: () => void;
  showRedact: boolean; onToggleRedact: () => void;
}) {
  const dark = useContext(DarkCtx);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function h(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  const menuBg = dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200";
  const itemCls = dark ? "text-gray-300 hover:bg-gray-700" : "text-gray-700 hover:bg-gray-50";
  const inputCls = dark ? "bg-gray-700 border-gray-600 text-gray-200" : "bg-white border-gray-300 text-gray-800";

  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen((o) => !o)}
        className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors ${
          open
            ? dark ? "bg-gray-700 border-gray-600 text-white" : "bg-gray-100 border-gray-300 text-gray-900"
            : dark ? "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700" : "bg-white border-gray-200 text-gray-700 hover:bg-gray-50"
        }`}>
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
        </svg>
        View Options
        <svg className="w-3 h-3 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" /></svg>
      </button>

      {open && (
        <div className={`absolute right-0 top-full mt-1 w-52 rounded-xl border shadow-xl z-40 overflow-hidden ${menuBg}`}>
          <div className="p-1">
            <button onClick={() => { onExpandAll(); setOpen(false); }}
              className={`flex w-full items-center gap-2 px-3 py-2 text-xs rounded-lg transition-colors ${itemCls}`}>
              <svg className="w-3.5 h-3.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" /></svg>
              Expand All
            </button>
            <button onClick={() => { onCollapseAll(); setOpen(false); }}
              className={`flex w-full items-center gap-2 px-3 py-2 text-xs rounded-lg transition-colors ${itemCls}`}>
              <svg className="w-3.5 h-3.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 9V4.5M9 9H4.5M9 9L3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5l5.25 5.25" /></svg>
              Collapse All
            </button>
          </div>
          <div className={`border-t ${dark ? "border-gray-700" : "border-gray-100"} p-1`}>
            <button onClick={() => { onTogglePageNav(); setOpen(false); }}
              className={`flex w-full items-center justify-between gap-2 px-3 py-2 text-xs rounded-lg transition-colors ${itemCls}`}>
              <span className="flex items-center gap-2">
                <svg className="w-3.5 h-3.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" /></svg>
                Page Navigator
              </span>
              {showPageNav && <svg className="w-3 h-3 text-blue-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" /></svg>}
            </button>
          </div>
          <div className={`border-t ${dark ? "border-gray-700" : "border-gray-100"} px-3 py-2`}>
            <p className={`text-xs mb-2 ${dark ? "text-gray-500" : "text-gray-400"}`}>Page range</p>
            <div className="flex items-center gap-2">
              <input type="number" min={1} max={totalPages} value={fromPage}
                onChange={(e) => onFromChange(Math.max(1, Math.min(+e.target.value, toPage)))}
                className={`w-16 rounded-md border px-2 py-1 text-xs text-center focus:outline-none focus:ring-1 focus:ring-blue-500 ${inputCls}`} />
              <span className={`text-xs ${dark ? "text-gray-500" : "text-gray-400"}`}>–</span>
              <input type="number" min={fromPage} max={totalPages} value={toPage}
                onChange={(e) => onToChange(Math.max(fromPage, Math.min(+e.target.value, totalPages)))}
                className={`w-16 rounded-md border px-2 py-1 text-xs text-center focus:outline-none focus:ring-1 focus:ring-blue-500 ${inputCls}`} />
            </div>
          </div>
          <div className={`border-t ${dark ? "border-gray-700" : "border-gray-100"} p-1`}>
            <button onClick={() => { onToggleStats(); setOpen(false); }}
              className={`flex w-full items-center justify-between gap-2 px-3 py-2 text-xs rounded-lg transition-colors ${itemCls}`}>
              <span className="flex items-center gap-2">
                <svg className="w-3.5 h-3.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" /></svg>
                Text Statistics
              </span>
              {showStats && <svg className="w-3 h-3 text-blue-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" /></svg>}
            </button>
            <button onClick={() => { onToggleRedact(); setOpen(false); }}
              className={`flex w-full items-center justify-between gap-2 px-3 py-2 text-xs rounded-lg transition-colors ${itemCls}`}>
              <span className="flex items-center gap-2">
                <svg className="w-3.5 h-3.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" /></svg>
                Redact Terms
              </span>
              {showRedact && <svg className="w-3 h-3 text-blue-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" /></svg>}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── DownloadMenu ─────────────────────────────────────────────────────────────

function DownloadMenu({ result, pages, getEff, onCopyMarkdown }: {
  result: ExtractResult; pages: Page[]; getEff: (p: Page) => string; onCopyMarkdown: () => void;
}) {
  const dark = useContext(DarkCtx);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const base = result.filename.replace(/\.[^.]+$/, "");

  useEffect(() => {
    function h(e: MouseEvent) { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); }
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  const opts = result.kind === "pdf" ? [
    { label: "Plain text (.txt)", ext: "txt", mime: "text/plain", build: () => buildTxt(pages, getEff) },
    { label: "JSON (.json)", ext: "json", mime: "application/json", build: () => buildJson(result, pages, getEff) },
    { label: "Markdown (.md)", ext: "md", mime: "text/markdown", build: () => buildMd(result.filename, pages, getEff) },
  ] : [
    { label: "Plain text (.txt)", ext: "txt", mime: "text/plain", build: () => (result as ImageResult).text },
  ];

  const menuBg = dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200";
  const itemCls = dark ? "text-gray-300 hover:bg-gray-700" : "text-gray-700 hover:bg-gray-50";

  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen((o) => !o)}
        className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors ${
          open
            ? dark ? "bg-gray-700 border-gray-600 text-white" : "bg-gray-100 border-gray-300 text-gray-900"
            : dark ? "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700" : "bg-white border-gray-200 text-gray-700 hover:bg-gray-50"
        }`}>
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
        </svg>
        Download
        <svg className="w-3 h-3 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" /></svg>
      </button>
      {open && (
        <div className={`absolute left-0 top-full mt-1 w-48 rounded-xl border shadow-xl z-40 overflow-hidden ${menuBg}`}>
          <div className="p-1">
            {opts.map((o) => (
              <button key={o.ext}
                onClick={() => { triggerDownload(o.build(), `${base}.${o.ext}`, o.mime); setOpen(false); }}
                className={`flex w-full items-center gap-2.5 px-3 py-2 text-xs rounded-lg transition-colors ${itemCls}`}>
                <span className={`w-8 text-center rounded font-mono text-xs font-semibold py-0.5 ${dark ? "bg-gray-700 text-gray-400" : "bg-gray-100 text-gray-500"}`}>.{o.ext}</span>
                {o.label.split("(")[0].trim()}
              </button>
            ))}
            {result.kind === "pdf" && (
              <>
                <div className={`my-1 border-t ${dark ? "border-gray-700" : "border-gray-100"}`} />
                <button onClick={() => { onCopyMarkdown(); setOpen(false); }}
                  className={`flex w-full items-center gap-2.5 px-3 py-2 text-xs rounded-lg transition-colors ${itemCls}`}>
                  <span className={`w-8 text-center rounded font-mono text-xs font-semibold py-0.5 ${dark ? "bg-gray-700 text-gray-400" : "bg-gray-100 text-gray-500"}`}>MD</span>
                  Copy as Markdown
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
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
    <div className={`rounded-xl border p-5 space-y-5 ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
      <div className="grid grid-cols-3 gap-4">
        {[
          { l: "Avg words / page", v: fmt(avgWords) },
          { l: "Shortest page", v: sorted[0] ? `Page ${sorted[0].page_number}` : "—" },
          { l: "Longest page", v: sorted[sorted.length - 1] ? `Page ${sorted[sorted.length - 1].page_number}` : "—" },
        ].map(({ l, v }) => (
          <div key={l}>
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
              <div className={`flex-1 rounded-full h-1.5 ${dark ? "bg-gray-700" : "bg-gray-100"}`}>
                <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: `${(count / maxCount) * 100}%` }} />
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
    <div className={`rounded-xl border p-4 space-y-3 ${dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
      <p className={`text-xs font-semibold uppercase tracking-wide ${dark ? "text-gray-500" : "text-gray-400"}`}>Redact Terms</p>
      <div className="flex gap-2">
        <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && add()}
          placeholder="Word or phrase to redact…"
          className={`flex-1 rounded-lg border px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-red-500 ${dark ? "bg-gray-700 border-gray-600 text-gray-100 placeholder-gray-500" : "bg-white border-gray-300 text-gray-900 placeholder-gray-400"}`} />
        <button onClick={add} className="rounded-lg bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-700 transition-colors">Add</button>
      </div>
      {terms.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {terms.map((t, i) => (
            <span key={i} className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium ${dark ? "bg-red-900/40 text-red-300" : "bg-red-50 text-red-700 border border-red-200"}`}>
              {t}<button onClick={() => onRemove(i)} className="ml-0.5 opacity-60 hover:opacity-100">×</button>
            </span>
          ))}
        </div>
      )}
      <p className={`text-xs ${dark ? "text-gray-600" : "text-gray-400"}`}>Terms appear as ████ in view and all downloads.</p>
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
      <div className={`px-4 py-2 rounded-t-xl border border-b-0 text-xs font-semibold uppercase tracking-wide ${dark ? "bg-gray-800 border-gray-700 text-gray-500" : "bg-gray-50 border-gray-200 text-gray-400"}`}>{label}</div>
      <div className={`rounded-b-xl border overflow-auto max-h-[72vh] ${dark ? "bg-gray-900 border-gray-700" : "bg-white border-gray-200"}`}>
        {(status === "idle" || status === "error") && (
          <div className="flex flex-col items-center justify-center h-48 gap-3 cursor-pointer p-6" onClick={() => inputRef.current?.click()}>
            <input ref={inputRef} type="file" accept={ACCEPTED_EXTS.join(",")} className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) processFile(f); e.target.value = ""; }} />
            <svg className="w-8 h-8 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
            <p className={`text-sm ${dark ? "text-gray-600" : "text-gray-400"}`}>Click to upload</p>
            {status === "error" && <p className="text-xs text-red-500">{err}</p>}
          </div>
        )}
        {status === "loading" && (
          <div className="flex items-center justify-center h-48">
            <svg className="w-5 h-5 text-blue-500 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
          </div>
        )}
        {status === "success" && result && (
          <div className="px-6 py-4 space-y-5">
            <p className={`text-xs font-medium ${dark ? "text-gray-500" : "text-gray-400"}`}>{result.filename}</p>
            {result.kind === "pdf" ? result.pages.map((p) => (
              <div key={p.page_number}>
                <div className={`flex items-center gap-3 py-2`}>
                  <div className={`flex-1 h-px ${dark ? "bg-gray-800" : "bg-gray-100"}`} />
                  <span className={`text-xs ${dark ? "text-gray-600" : "text-gray-400"}`}>Page {p.page_number}</span>
                  <div className={`flex-1 h-px ${dark ? "bg-gray-800" : "bg-gray-100"}`} />
                </div>
                <pre className={`text-xs whitespace-pre-wrap font-sans leading-relaxed ${dark ? "text-gray-300" : "text-gray-700"}`}>{p.text || "—"}</pre>
              </div>
            )) : (
              <pre className={`text-xs whitespace-pre-wrap font-sans leading-relaxed ${dark ? "text-gray-300" : "text-gray-700"}`}>{(result as ImageResult).text || "—"}</pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── DocumentPage ─────────────────────────────────────────────────────────────

interface DocumentPageProps {
  page: Page;
  effectiveText: string;
  regex: RegExp | null;
  currentLocalIdx: number;
  collapsed: boolean;
  onToggleCollapse: () => void;
  isBookmarked: boolean;
  onToggleBookmark: () => void;
  isDragging: boolean;
  onDragStart: () => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragEnd: () => void;
}

function DocumentPage({
  page, effectiveText, regex, currentLocalIdx,
  collapsed, onToggleCollapse,
  isBookmarked, onToggleBookmark,
  isDragging, onDragStart, onDragOver, onDragEnd,
}: DocumentPageProps) {
  const dark = useContext(DarkCtx);
  const [pageCopied, setPageCopied] = useState(false);
  const hasTable = useMemo(() => detectTable(page.text), [page.text]);

  function copyPage() {
    navigator.clipboard.writeText(effectiveText).then(() => {
      setPageCopied(true); setTimeout(() => setPageCopied(false), 2000);
    });
  }

  const ruleCls = dark ? "bg-gray-800" : "bg-gray-200";
  const labelCls = dark ? "text-gray-600 hover:text-gray-400" : "text-gray-400 hover:text-gray-600";

  return (
    <div id={`page-${page.page_number}`} draggable
      onDragStart={onDragStart} onDragOver={onDragOver} onDragEnd={onDragEnd}
      className={`scroll-mt-16 transition-opacity ${isDragging ? "opacity-30" : ""}`}>

      {/* ── Page divider ── */}
      <div className="group flex items-center gap-3 py-5">
        <div className={`flex-1 h-px ${ruleCls}`} />

        <div className="flex items-center gap-2 select-none">
          {/* Collapse toggle */}
          <button onClick={onToggleCollapse}
            className={`flex items-center gap-1.5 text-xs font-medium transition-colors ${labelCls}`}>
            {collapsed && (
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
              </svg>
            )}
            Page {page.page_number}
          </button>

          {/* Inline badges */}
          {page.method === "ocr" && (
            <span className={`text-xs font-medium ${dark ? "text-amber-500/70" : "text-amber-400"}`}>OCR</span>
          )}
          {hasTable && (
            <span className={`text-xs font-medium ${dark ? "text-purple-500/70" : "text-purple-400"}`}>Table</span>
          )}

          {/* Hover actions */}
          <div className="opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1.5">
            <button onClick={onToggleBookmark} title={isBookmarked ? "Remove bookmark" : "Bookmark"}
              className={`transition-colors ${isBookmarked ? "text-amber-500" : dark ? "text-gray-700 hover:text-amber-400" : "text-gray-300 hover:text-amber-500"}`}>
              <svg className="w-3.5 h-3.5" fill={isBookmarked ? "currentColor" : "none"} viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0111.186 0z" />
              </svg>
            </button>
            <button onClick={copyPage} title="Copy page"
              className={`transition-colors ${dark ? "text-gray-700 hover:text-gray-400" : "text-gray-300 hover:text-gray-500"}`}>
              {pageCopied
                ? <svg className="w-3.5 h-3.5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                : <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184" /></svg>}
            </button>
            <svg className={`w-3.5 h-3.5 cursor-grab ${dark ? "text-gray-700" : "text-gray-300"}`} fill="currentColor" viewBox="0 0 20 20">
              <path d="M7 2a2 2 0 1 0 .001 4.001A2 2 0 0 0 7 2zm0 6a2 2 0 1 0 .001 4.001A2 2 0 0 0 7 8zm0 6a2 2 0 1 0 .001 4.001A2 2 0 0 0 7 14zm6-8a2 2 0 1 0-.001-4.001A2 2 0 0 0 13 6zm0 2a2 2 0 1 0 .001 4.001A2 2 0 0 0 13 8zm0 6a2 2 0 1 0 .001 4.001A2 2 0 0 0 13 14z" />
            </svg>
          </div>
        </div>

        <div className={`flex-1 h-px ${ruleCls}`} />
      </div>

      {/* ── Text content ── */}
      {!collapsed && (
        <div className="pb-2">
          {effectiveText.trim() ? (
            <p className={`text-[15px] leading-[1.75] whitespace-pre-wrap font-sans ${dark ? "text-gray-200" : "text-gray-800"}`}>
              {highlightText(effectiveText, regex, currentLocalIdx)}
            </p>
          ) : (
            <p className={`text-sm italic py-2 ${dark ? "text-gray-700" : "text-gray-400"}`}>No text found on this page.</p>
          )}
        </div>
      )}
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
  const [searchExpanded, setSearchExpanded] = useState(false);
  const [query, setQuery] = useState("");
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [wholeWord, setWholeWord] = useState(false);
  const [currentMatchIdx, setCurrentMatchIdx] = useState(0);
  const [showReplace, setShowReplace] = useState(false);
  const [replaceWith, setReplaceWith] = useState("");
  const [modifiedTexts, setModifiedTexts] = useState<Map<number, string>>(new Map());
  const searchInputRef = useRef<HTMLInputElement>(null);

  // View options
  const [showPageNav, setShowPageNav] = useState(false);
  const [fromPage, setFromPage] = useState(1);
  const [toPage, setToPage] = useState(1);
  const [showStats, setShowStats] = useState(false);
  const [showRedact, setShowRedact] = useState(false);
  const [redactTerms, setRedactTerms] = useState<string[]>([]);

  // Info panel
  const [showDetails, setShowDetails] = useState(false);

  // Copy
  const [copied, setCopied] = useState(false);
  const [showGoTop, setShowGoTop] = useState(false);

  // Merge mode
  const [mergedSections, setMergedSections] = useState<MergeSection[]>([]);
  const [mergeStatus, setMergeStatus] = useState<Status>("idle");
  const [mergeError, setMergeError] = useState("");
  const mergeInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const filenameRowRef = useRef<HTMLDivElement>(null);

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
      setSearchExpanded(false); setQuery("");
    }
  }, [result]);
  useEffect(() => { setCurrentMatchIdx(0); }, [query, caseSensitive, wholeWord, fromPage, toPage]);
  useEffect(() => {
    if (searchExpanded) setTimeout(() => searchInputRef.current?.focus(), 50);
  }, [searchExpanded]);

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
    setTimeout(() => document.getElementById(`page-${match.pageNum}`)?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
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
    setStatus("idle"); setResult(null); setErrorMsg(""); setQuery(""); setSearchExpanded(false);
    setPageOrder([]); setCollapsedPages(new Set()); setBookmarkedPages(new Set());
    setModifiedTexts(new Map()); setRedactTerms([]); setShowDetails(false);
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
  const pageBg = dark ? "bg-gray-950 text-gray-100" : "bg-white text-gray-900";
  const navBg = dark ? "bg-gray-900 border-gray-800" : "bg-white border-gray-200";
  const toolbarBg = dark ? "border-gray-800" : "border-gray-200";

  return (
    <DarkCtx.Provider value={darkMode}>
      <div className={`min-h-screen ${pageBg}`}>

        {/* ── Navbar ── */}
        <header className={`border-b sticky top-0 z-20 ${navBg}`}>
          <div className="max-w-4xl mx-auto px-6 h-12 flex items-center gap-4">
            <span className={`text-sm font-semibold tracking-tight ${dark ? "text-gray-200" : "text-gray-900"}`}>doc-to-text</span>
            <div className="flex items-center gap-0.5 ml-2">
              {(["single", "compare", "merge"] as AppMode[]).map((m) => (
                <button key={m} onClick={() => setAppMode(m)}
                  className={`px-3 py-1 rounded-md text-xs font-medium transition-colors capitalize ${appMode === m
                    ? dark ? "bg-gray-700 text-gray-100" : "bg-gray-100 text-gray-900"
                    : dark ? "text-gray-500 hover:text-gray-300" : "text-gray-400 hover:text-gray-700"}`}>
                  {m}
                </button>
              ))}
            </div>
            <div className="ml-auto flex items-center gap-2">
              {status === "success" && appMode === "single" && (
                <button onClick={reset} className={`text-xs transition-colors ${dark ? "text-gray-600 hover:text-gray-400" : "text-gray-400 hover:text-gray-600"}`}>
                  ← New file
                </button>
              )}
              {/* Dark mode icon */}
              <button onClick={() => setDarkMode((d) => !d)} title={dark ? "Light mode" : "Dark mode"}
                className={`p-1.5 rounded-md transition-colors ${dark ? "text-gray-500 hover:text-yellow-400 hover:bg-gray-800" : "text-gray-400 hover:text-gray-700 hover:bg-gray-100"}`}>
                {dark
                  ? <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" /></svg>
                  : <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" /></svg>}
              </button>
            </div>
          </div>
        </header>

        <main className="max-w-4xl mx-auto px-6">

          {/* ══ COMPARE ══ */}
          {appMode === "compare" && (
            <div className="grid grid-cols-2 gap-4 py-8">
              <ComparePanel label="Document A" />
              <ComparePanel label="Document B" />
            </div>
          )}

          {/* ══ MERGE ══ */}
          {appMode === "merge" && (
            <div className="py-8 space-y-5">
              {mergeStatus !== "success" && (
                <div onClick={() => mergeInputRef.current?.click()}
                  className={`cursor-pointer rounded-2xl border-2 border-dashed flex flex-col items-center justify-center gap-3 py-16 text-center transition-colors ${dark ? "border-gray-800 hover:border-gray-700" : "border-gray-200 hover:border-gray-300"}`}>
                  <input ref={mergeInputRef} type="file" multiple accept={ACCEPTED_EXTS.join(",")} className="hidden"
                    onChange={(e) => { if (e.target.files?.length) processMergeFiles(e.target.files); e.target.value = ""; }} />
                  <svg className="w-8 h-8 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}><path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" /></svg>
                  <div>
                    <p className={`text-sm font-medium ${dark ? "text-gray-400" : "text-gray-600"}`}>Select multiple files to merge</p>
                    <p className={`text-xs mt-1 ${dark ? "text-gray-600" : "text-gray-400"}`}>PDF, PNG, JPG, JPEG, TIFF, BMP</p>
                  </div>
                  {mergeStatus === "error" && <p className="text-sm text-red-500">{mergeError}</p>}
                </div>
              )}
              {mergeStatus === "loading" && <div className="flex justify-center py-20"><svg className="w-6 h-6 text-blue-500 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" /></svg></div>}
              {mergeStatus === "success" && mergedSections.length > 0 && (
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    <p className={`text-sm font-medium ${dark ? "text-gray-300" : "text-gray-700"}`}>{mergedSections.length} files merged</p>
                    <button onClick={() => { setMergedSections([]); setMergeStatus("idle"); }} className={`text-xs ${dark ? "text-gray-600 hover:text-gray-400" : "text-gray-400 hover:text-gray-600"}`}>← Start over</button>
                    <button onClick={() => triggerDownload(buildMergeTxt(mergedSections), "merged.txt", "text/plain")}
                      className="ml-auto inline-flex items-center gap-2 rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700 transition-colors">
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" /></svg>
                      Download merged.txt
                    </button>
                  </div>
                  {mergedSections.map((section, si) => (
                    <div key={si}>
                      <p className={`text-xs font-semibold uppercase tracking-wide mb-3 ${dark ? "text-gray-500" : "text-gray-400"}`}>{section.filename}</p>
                      {section.pages.map((p) => (
                        <div key={p.page_number} className="mb-4">
                          <div className={`flex items-center gap-3 py-3`}>
                            <div className={`flex-1 h-px ${dark ? "bg-gray-800" : "bg-gray-100"}`} />
                            <span className={`text-xs ${dark ? "text-gray-600" : "text-gray-400"}`}>Page {p.page_number}</span>
                            <div className={`flex-1 h-px ${dark ? "bg-gray-800" : "bg-gray-100"}`} />
                          </div>
                          <p className={`text-sm whitespace-pre-wrap font-sans leading-relaxed ${dark ? "text-gray-300" : "text-gray-700"}`}>{p.text || "—"}</p>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ══ SINGLE ══ */}
          {appMode === "single" && (
            <>
              {/* Drop zone */}
              {(status === "idle" || status === "error") && (
                <div className="py-16">
                  <div onDrop={onFileDrop} onDragOver={onFileDragOver} onDragLeave={onFileDragLeave}
                    onClick={() => inputRef.current?.click()}
                    className={`cursor-pointer rounded-2xl border-2 border-dashed flex flex-col items-center justify-center gap-4 py-20 text-center transition-colors ${fileDragging ? "border-blue-400 bg-blue-50 dark:bg-blue-900/10" : dark ? "border-gray-800 hover:border-gray-700" : "border-gray-200 hover:border-gray-300"}`}>
                    <input ref={inputRef} type="file" accept={ACCEPTED_EXTS.join(",")} className="hidden" onChange={onFileChange} />
                    <svg className="w-10 h-10 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                    </svg>
                    <div>
                      <p className={`text-base font-medium ${dark ? "text-gray-400" : "text-gray-600"}`}>{fileDragging ? "Drop file here" : "Drop a file or click to browse"}</p>
                      <p className={`text-xs mt-1.5 ${dark ? "text-gray-600" : "text-gray-400"}`}>PDF · PNG · JPG · JPEG · TIFF · BMP</p>
                    </div>
                  </div>
                  {status === "error" && (
                    <p className="mt-4 text-sm text-red-500 text-center">{errorMsg}</p>
                  )}
                </div>
              )}

              {/* Loading */}
              {status === "loading" && (
                <div className="flex flex-col items-center justify-center py-40 gap-4">
                  <svg className="w-7 h-7 text-blue-500 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <p className={`text-sm ${dark ? "text-gray-600" : "text-gray-400"}`}>Extracting text…</p>
                </div>
              )}

              {/* Document view */}
              {status === "success" && result && (
                <div className="py-8 space-y-5">

                  {/* ── Filename ── */}
                  <div className="flex items-center gap-3 min-w-0 flex-wrap">
                    <h2 className={`text-lg font-semibold ${dark ? "text-gray-100" : "text-gray-900"}`}>{result.filename}</h2>
                    {result.kind === "pdf" && bookmarkedPages.size > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap">
                        {pagesInRange.filter((p) => bookmarkedPages.has(p.page_number)).map((p) => (
                          <button key={p.page_number} onClick={() => scrollToPage(p.page_number)}
                            className="inline-flex items-center gap-1 rounded-md bg-amber-100 dark:bg-amber-900/30 border border-amber-200 dark:border-amber-800/50 px-2 py-0.5 text-xs font-medium text-amber-700 dark:text-amber-400 hover:bg-amber-200 transition-colors">
                            <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 24 24"><path d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0111.186 0z" /></svg>
                            {p.page_number}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* ── Summary cards ── */}
                  {result.kind === "pdf" && (
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {([
                        ["Pages", result.total_pages.toString()],
                        ["Words", fmt(wordCount)],
                        ["Characters", fmt(result.total_characters)],
                        ["Extract time", `${result.extraction_time_seconds}s`],
                      ] as [string, string][]).map(([label, value]) => (
                        <div key={label} className={`rounded-xl border px-4 py-3 ${dark ? "bg-gray-800 border-gray-700" : "bg-gray-50 border-gray-200"}`}>
                          <p className={`text-xs ${dark ? "text-gray-500" : "text-gray-400"}`}>{label}</p>
                          <p className={`text-2xl font-bold mt-0.5 ${dark ? "text-white" : "text-gray-900"}`}>{value}</p>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* ── Search bar ── */}
                  {result.kind === "pdf" && (
                    <div className={`flex items-center gap-2 rounded-xl border px-3 py-2.5 shadow-sm ${dark ? "bg-gray-900 border-gray-800" : "bg-white border-gray-200"}`}>
                      <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 15.803a7.5 7.5 0 0010.607 0z" />
                      </svg>
                      <input ref={searchInputRef} type="text" value={query} onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); navigateToMatch(e.shiftKey ? safeMatchIdx - 1 : safeMatchIdx + 1); } if (e.key === "Escape") setQuery(""); }}
                        placeholder="Search text… (Enter = next, Shift+Enter = prev)"
                        className={`flex-1 bg-transparent text-sm focus:outline-none ${dark ? "text-gray-200 placeholder-gray-600" : "text-gray-800 placeholder-gray-400"}`} />
                      {query && (
                        <button onClick={() => setQuery("")} className={`flex-shrink-0 ${dark ? "text-gray-600 hover:text-gray-400" : "text-gray-300 hover:text-gray-500"}`}>
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                        </button>
                      )}
                      {query.trim() && (
                        <>
                          <button onClick={() => navigateToMatch(safeMatchIdx - 1)} disabled={!allMatches.length}
                            className={`p-1 rounded transition-colors disabled:opacity-30 ${dark ? "text-gray-500 hover:text-gray-300 hover:bg-gray-800" : "text-gray-400 hover:text-gray-700 hover:bg-gray-100"}`}>
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}><path strokeLinecap="round" strokeLinejoin="round" d="M4.5 15.75l7.5-7.5 7.5 7.5" /></svg>
                          </button>
                          <button onClick={() => navigateToMatch(safeMatchIdx + 1)} disabled={!allMatches.length}
                            className={`p-1 rounded transition-colors disabled:opacity-30 ${dark ? "text-gray-500 hover:text-gray-300 hover:bg-gray-800" : "text-gray-400 hover:text-gray-700 hover:bg-gray-100"}`}>
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" /></svg>
                          </button>
                          <span className={`text-xs tabular-nums flex-shrink-0 min-w-[3.5rem] text-center ${dark ? "text-gray-500" : "text-gray-400"}`}>
                            {allMatches.length === 0 ? "No results" : `${safeMatchIdx + 1} / ${allMatches.length}`}
                          </span>
                        </>
                      )}
                      <div className={`w-px h-5 flex-shrink-0 ${dark ? "bg-gray-700" : "bg-gray-200"}`} />
                      <TBtn active={caseSensitive} onClick={() => setCaseSensitive((v) => !v)} title="Match case">Aa</TBtn>
                      <TBtn active={wholeWord} onClick={() => setWholeWord((v) => !v)} title="Whole word">W</TBtn>
                      <TBtn active={showReplace} onClick={() => setShowReplace((v) => !v)} title="Find & Replace">↕</TBtn>
                    </div>
                  )}

                  {/* Replace bar */}
                  {result.kind === "pdf" && showReplace && (
                    <div className={`flex items-center gap-2 rounded-xl border px-3 py-2.5 shadow-sm ${dark ? "bg-gray-900 border-gray-800" : "bg-white border-gray-200"}`}>
                      <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" /></svg>
                      <input type="text" value={replaceWith} onChange={(e) => setReplaceWith(e.target.value)}
                        placeholder="Replace with…"
                        className={`flex-1 bg-transparent text-sm focus:outline-none ${dark ? "text-gray-200 placeholder-gray-600" : "text-gray-800 placeholder-gray-400"}`} />
                      <button onClick={replaceCurrent} disabled={!currentMatch}
                        className={`text-xs px-2.5 py-1 rounded-lg border transition-colors disabled:opacity-40 ${dark ? "border-gray-700 text-gray-400 hover:bg-gray-800" : "border-gray-200 text-gray-600 hover:bg-gray-50"}`}>Replace</button>
                      <button onClick={replaceAll} disabled={!allMatches.length}
                        className="text-xs px-2.5 py-1 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-40">Replace All</button>
                      {modifiedTexts.size > 0 && (
                        <>
                          <span className={`text-xs ${dark ? "text-gray-600" : "text-gray-400"}`}>{modifiedTexts.size} page{modifiedTexts.size !== 1 ? "s" : ""} modified</span>
                          <button onClick={() => setModifiedTexts(new Map())} className="text-xs text-red-400 hover:text-red-500">Reset</button>
                        </>
                      )}
                    </div>
                  )}

                  {/* ── Actions toolbar ── */}
                  <div className={`flex items-center gap-2 flex-wrap border-y py-2.5 ${dark ? "border-gray-800" : "border-gray-100"}`}>
                    <button onClick={copyAll}
                      className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors ${dark ? "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700" : "bg-white border-gray-200 text-gray-700 hover:bg-gray-50"}`}>
                      {copied
                        ? <svg className="w-4 h-4 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                        : <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184" /></svg>}
                      {copied ? "Copied!" : "Copy All"}
                    </button>

                    <DownloadMenu result={result} pages={downloadPages} getEff={getEff} onCopyMarkdown={copyAsMarkdown} />

                    <button onClick={handlePrint}
                      className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors ${dark ? "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700" : "bg-white border-gray-200 text-gray-700 hover:bg-gray-50"}`}>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M6.72 13.829c-.24.03-.48.062-.72.096m.72-.096a42.415 42.415 0 0110.56 0m-10.56 0L6.34 18m10.94-4.171c.24.03.48.062.72.096m-.72-.096L17.66 18m0 0l.229 2.523a1.125 1.125 0 01-1.12 1.227H7.231c-.662 0-1.18-.568-1.12-1.227L6.34 18m11.318 0h1.091A2.25 2.25 0 0021 15.75V9.456c0-1.081-.768-2.015-1.837-2.175a48.055 48.055 0 00-1.913-.247M6.34 18H5.25A2.25 2.25 0 013 15.75V9.456c0-1.081.768-2.015 1.837-2.175a48.041 48.041 0 011.913-.247m10.5 0a48.536 48.536 0 00-10.5 0m10.5 0V3.375c0-.621-.504-1.125-1.125-1.125h-8.25c-.621 0-1.125.504-1.125 1.125v3.659M18 10.5h.008v.008H18V10.5zm-3 0h.008v.008H15V10.5z" /></svg>
                      Print
                    </button>

                    {result.kind === "pdf" && (
                      <button onClick={exportSummary}
                        className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors ${dark ? "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700" : "bg-white border-gray-200 text-gray-700 hover:bg-gray-50"}`}>
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25z" /></svg>
                        Export Summary
                      </button>
                    )}

                    {result.kind === "pdf" && (
                      <ViewOptionsMenu
                        showPageNav={showPageNav} onTogglePageNav={() => setShowPageNav((v) => !v)}
                        onExpandAll={expandAll} onCollapseAll={collapseAll}
                        fromPage={fromPage} toPage={toPage} totalPages={result.total_pages}
                        onFromChange={setFromPage} onToChange={setToPage}
                        showStats={showStats} onToggleStats={() => setShowStats((v) => !v)}
                        showRedact={showRedact} onToggleRedact={() => setShowRedact((v) => !v)}
                      />
                    )}
                  </div>

                  {/* Page navigator */}
                  {showPageNav && result.kind === "pdf" && (
                    <div className={`flex flex-wrap gap-1.5 p-3 rounded-xl border ${dark ? "bg-gray-900 border-gray-800" : "bg-gray-50 border-gray-200"}`}>
                      {orderedPages.map((p) => (
                        <button key={p.page_number} onClick={() => scrollToPage(p.page_number)}
                          className={`relative rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                            p.page_number < fromPage || p.page_number > toPage
                              ? dark ? "text-gray-800" : "text-gray-300"
                              : dark ? "text-gray-500 hover:text-gray-300 hover:bg-gray-800" : "text-gray-500 hover:text-gray-800 hover:bg-white border border-transparent hover:border-gray-200"
                          }`}>
                          {p.page_number}
                          {bookmarkedPages.has(p.page_number) && (
                            <span className="absolute top-0.5 right-0.5 w-1.5 h-1.5 bg-amber-400 rounded-full" />
                          )}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Stats / Redact panels */}
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
                    <div>
                      {displayPages.map((page, idx) => (
                        <DocumentPage key={page.page_number} page={page}
                          effectiveText={getEff(page)}
                          regex={searchRegex}
                          currentLocalIdx={currentMatch?.pageNum === page.page_number ? currentMatch.localIdx : -1}
                          collapsed={collapsedPages.has(page.page_number)}
                          onToggleCollapse={() => setCollapsedPages((prev) => { const n = new Set(prev); n.has(page.page_number) ? n.delete(page.page_number) : n.add(page.page_number); return n; })}
                          isBookmarked={bookmarkedPages.has(page.page_number)}
                          onToggleBookmark={() => setBookmarkedPages((prev) => { const n = new Set(prev); n.has(page.page_number) ? n.delete(page.page_number) : n.add(page.page_number); return n; })}
                          isDragging={draggingPageNum === page.page_number}
                          onDragStart={() => handlePageDragStart(idx)}
                          onDragOver={(e) => handlePageDragOver(e, idx)}
                          onDragEnd={handlePageDragEnd}
                        />
                      ))}
                      {displayPages.length === 0 && searchRegex && (
                        <p className={`text-sm py-16 text-center ${dark ? "text-gray-700" : "text-gray-400"}`}>No pages match your search.</p>
                      )}
                    </div>
                  ) : (
                    /* Image result */
                    <div className="pt-2">
                      <p className={`text-[15px] leading-[1.75] whitespace-pre-wrap font-sans ${dark ? "text-gray-200" : "text-gray-800"}`}>
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
            className={`fixed bottom-6 right-6 z-30 rounded-full p-2.5 shadow-lg transition-colors ${dark ? "bg-gray-800 text-gray-400 hover:text-gray-200 border border-gray-700" : "bg-white text-gray-500 hover:text-gray-800 border border-gray-200 shadow-md"}`}
            title="Back to top">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4.5 15.75l7.5-7.5 7.5 7.5" /></svg>
          </button>
        )}
      </div>
    </DarkCtx.Provider>
  );
}
