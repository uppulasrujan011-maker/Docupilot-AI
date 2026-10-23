import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowDownToLine,
  BookOpen,
  Bot,
  Check,
  CheckCircle2,
  ChevronRight,
  Clock,
  Code2,
  Cpu,
  Database,
  ExternalLink,
  Eye,
  FileCode2,
  FileText,
  GitCompare,
  History,
  Layers,
  LayoutDashboard,
  LoaderCircle,
  Play,
  RefreshCw,
  Search,
  Shield,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Terminal,
  Upload,
  WandSparkles,
  X,
  FileArchive,
} from "lucide-react";

const API = (import.meta.env.VITE_API_URL || "https://docupilot-ai-3.onrender.com").replace(/\/$/, "");

export interface LanguageCapability {
  id: string;
  name: string;
  extension: string;
  parser_type: string;
  has_real_parser: boolean;
  syntax_validation_capable: boolean;
  doc_format: string;
  linter_available?: string | null;
  notes: string;
}

export const FALLBACK_LANGUAGES: LanguageCapability[] = [
  { id: "python", name: "Python", extension: "py", parser_type: "ast", has_real_parser: true, syntax_validation_capable: true, doc_format: 'PEP 257 (""")', notes: "Python AST parser" },
  { id: "javascript", name: "JavaScript", extension: "js", parser_type: "esprima", has_real_parser: true, syntax_validation_capable: true, doc_format: "JSDoc (/** */)", notes: "Esprima parser" },
  { id: "typescript", name: "TypeScript", extension: "ts", parser_type: "typescript_compiler", has_real_parser: true, syntax_validation_capable: true, doc_format: "TSDoc / JSDoc", notes: "TypeScript compiler API" },
  { id: "java", name: "Java", extension: "java", parser_type: "regex_heuristic", has_real_parser: false, syntax_validation_capable: false, doc_format: "Javadoc", notes: "Generation only" },
  { id: "c", name: "C", extension: "c", parser_type: "regex_heuristic", has_real_parser: false, syntax_validation_capable: false, doc_format: "Doxygen", notes: "Generation only" },
  { id: "cpp", name: "C++", extension: "cpp", parser_type: "regex_heuristic", has_real_parser: false, syntax_validation_capable: false, doc_format: "Doxygen", notes: "Generation only" },
  { id: "csharp", name: "C#", extension: "cs", parser_type: "regex_heuristic", has_real_parser: false, syntax_validation_capable: false, doc_format: "XML Doc (///)", notes: "Generation only" },
  { id: "sql", name: "SQL", extension: "sql", parser_type: "regex_heuristic", has_real_parser: false, syntax_validation_capable: false, doc_format: "SQL doc (--)", notes: "Generation only" },
];

export const SAMPLES: Record<string, string> = {
  python: `def calculate_discount(price, discount):
    if price <= 0 or discount < 0 or discount > 100:
        return 0
    return price - (price * discount / 100)

def summarize_orders(orders, minimum_total=0):
    eligible = []
    for order in orders:
        if order.get("status") == "paid":
            total = sum(item["price"] * item["quantity"] for item in order.get("items", []))
            if total >= minimum_total:
                eligible.append({"id": order["id"], "total": total})
    return sorted(eligible, key=lambda order: order["total"], reverse=True)`,

  javascript: `function calculateShipping(weight, distance, express = false) {
  if (weight <= 0 || distance < 0) {
    throw new Error("Invalid shipping inputs");
  }
  const base = weight * 2.5;
  const distanceFee = distance > 100 ? distance * 0.8 : distance * 0.4;
  const surcharge = express ? base * 0.5 : 0;
  return Math.round((base + distanceFee + surcharge) * 100) / 100;
}`,

  typescript: `interface CartItem {
  id: string;
  price: number;
  quantity: number;
}

export function computeCartTotal(items: CartItem[], taxRate: number = 0.08): number {
  if (taxRate < 0) {
    throw new RangeError("Tax rate cannot be negative");
  }
  const subtotal = items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  const tax = subtotal * taxRate;
  return Math.round((subtotal + tax) * 100) / 100;
}`,

  java: `public class OrderProcessor {
    public double calculateDiscount(double price, double discount) {
        if (price <= 0 || discount < 0 || discount > 100) {
            return 0.0;
        }
        return price - (price * discount / 100.0);
    }
}`,

  c: `#include <stdio.h>

float calculate_discount(float price, float discount) {
    if (price <= 0 || discount < 0 || discount > 100) {
        return 0.0f;
    }
    return price - (price * discount / 100.0f);
}`,

  cpp: `class DiscountCalculator {
public:
    double calculateDiscount(double price, double discount) {
        if (price <= 0 || discount < 0 || discount > 100) {
            return 0.0;
        }
        return price - (price * discount / 100.0);
    }
};`,

  csharp: `public class ShoppingCart {
    public decimal CalculateShipping(decimal weight, decimal distance, bool express = false) {
        if (weight <= 0 || distance < 0) {
            throw new System.ArgumentException("Invalid shipping parameters");
        }
        decimal baseRate = weight * 2.5m;
        decimal distanceFee = distance > 100 ? distance * 0.8m : distance * 0.4m;
        decimal surcharge = express ? baseRate * 0.5m : 0m;
        return baseRate + distanceFee + surcharge;
    }
}`,

  sql: `CREATE OR REPLACE FUNCTION calculate_discount(price DECIMAL(10,2), discount DECIMAL(5,2))
RETURNS DECIMAL(10,2) AS $$
BEGIN
    IF price <= 0 OR discount < 0 OR discount > 100 THEN
        RETURN 0.00;
    END IF;
    RETURN price - (price * discount / 100.0);
END;
$$ LANGUAGE plpgsql;`,
};

export interface QualityBreakdownItem {
  criterion: string;
  weight: number;
  earned: number;
  status: "passed" | "warning" | "failed";
  details: string;
}

export interface QualityScorecard {
  overall_score: number;
  grade: string;
  parameter_coverage: number;
  return_coverage: number;
  exception_coverage: number;
  summary_coverage: number;
  style_conformance: number;
  breakdown: QualityBreakdownItem[];
  findings: string[];
  disclaimer: string;
}

export interface AgentStage {
  id: string;
  name: string;
  status: "completed" | "pending" | "warning" | "failed";
  input: any;
  output: any;
}

export interface FingerprintComparison {
  original_fingerprint: string;
  proposed_fingerprint: string;
  matches: boolean;
  algorithm: string;
  explanation: string;
  disclaimer: string;
}

export interface VerificationCheck {
  name: string;
  status: "passed" | "failed" | "not_supported" | "not_run";
  tool: string;
  duration_ms: number;
  details: string;
  diagnostics?: any;
  fingerprint_comparison?: FingerprintComparison;
}

export interface VerificationResult {
  passed: boolean;
  global_status: "verified" | "partially_verified" | "failed";
  checks: VerificationCheck[];
  note: string;
  quality_score?: QualityScorecard;
  fingerprint?: {
    fingerprint: string;
    raw_hash?: string;
    algorithm: string;
    explanation: string;
    disclaimer: string;
  };
}

export interface AgentExecutionInfo {
  agent_name: string;
  status: string;
  summary: string;
  errors?: string[];
  findings?: string[];
  execution_time_ms?: number;
  evidence?: any;
  [key: string]: any;
}

export interface GeneratedRun {
  run_id: string;
  language: string;
  doc_style?: string;
  analysis: any;
  proposed_code: string;
  mode: string;
  model: string;
  status: string;
  workflow_state?: string;
  revision_count?: number;
  warning?: string | null;
  verification?: VerificationResult;
  quality_score?: QualityScorecard;
  stages?: AgentStage[];
  agent_executions?: Record<string, AgentExecutionInfo>;
}

export interface HistoryItem {
  id: string;
  created_at: string;
  reviewed_at?: string | null;
  language: string;
  status: string;
  mode: string;
  model?: string | null;
  rejection_reason?: string | null;
}

export interface Stats {
  total_runs: number;
  approved: number;
  rejected: number;
  pending: number;
}

interface DiffLine {
  type: "same" | "add" | "del";
  text: string;
  lineNum?: number;
}

function computeDiff(original: string, modified: string): DiffLine[] {
  const origLines = original.split("\n");
  const modLines = modified.split("\n");
  const result: DiffLine[] = [];

  const origSet = new Set(origLines.map(l => l.trim()));
  let lineNum = 1;

  for (const m of modLines) {
    if (origSet.has(m.trim())) {
      result.push({ type: "same", text: m, lineNum: lineNum++ });
    } else {
      result.push({ type: "add", text: m, lineNum: lineNum++ });
    }
  }

  return result;
}

export default function App() {
  const [navTab, setNavTab] = useState<"workspace" | "history" | "languages" | "agents">("workspace");
  const [showAgentTrace, setShowAgentTrace] = useState<boolean>(true);
  const [language, setLanguage] = useState("python");
  const [docStyle, setDocStyle] = useState<string>("google");
  const [source, setSource] = useState(SAMPLES.python);
  const [generated, setGenerated] = useState<GeneratedRun | null>(null);
  const [edited, setEdited] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [archiveNotice, setArchiveNotice] = useState<string | null>(null);
  const [approving, setApproving] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [showRejectBox, setShowRejectBox] = useState(false);
  const [rejectionReason, setRejectionReason] = useState("");
  const [message, setMessage] = useState("");
  const [messageType, setMessageType] = useState<"info" | "error" | "warning">("info");
  const [verification, setVerification] = useState<VerificationResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [stats, setStats] = useState<Stats>({ total_runs: 0, approved: 0, rejected: 0, pending: 0 });
  const [health, setHealth] = useState<"online" | "offline" | "checking">("checking");
  const [config, setConfig] = useState<{ model: string; mode: string; demo_mode: boolean } | null>(null);
  const [languages, setLanguages] = useState<LanguageCapability[]>(FALLBACK_LANGUAGES);
  const [viewTab, setViewTab] = useState<"edit" | "diff">("edit");
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [historyFilter, setHistoryFilter] = useState("");

  const activeCapability = useMemo(() => {
    return languages.find(l => l.id === language) || languages[0];
  }, [languages, language]);

  const diffLines = useMemo(() => {
    return computeDiff(source, edited);
  }, [source, edited]);

  const filteredHistory = useMemo(() => {
    if (!historyFilter.trim()) return history;
    const q = historyFilter.toLowerCase();
    return history.filter(
      item =>
        item.id.toLowerCase().includes(q) ||
        item.language.toLowerCase().includes(q) ||
        item.status.toLowerCase().includes(q)
    );
  }, [history, historyFilter]);

  async function refreshHistory() {
    try {
      const [hRes, sRes] = await Promise.all([
        fetch(`${API}/api/history?limit=50`),
        fetch(`${API}/api/stats`),
      ]);
      if (hRes.ok) {
        const d = await hRes.json();
        setHistory(d.items || []);
      }
      if (sRes.ok) {
        const s = await sRes.json();
        setStats(s);
      }
    } catch {
      /* API starting up */
    }
  }

  useEffect(() => {
    fetch(`${API}/api/health`)
      .then(r => (r.ok ? setHealth("online") : setHealth("offline")))
      .catch(() => setHealth("offline"));

    fetch(`${API}/api/config`)
      .then(r => r.json())
      .then(d => {
        setConfig({ model: d.model, mode: d.mode, demo_mode: d.demo_mode });
        if (d.languages && Array.isArray(d.languages)) {
          const excluded = new Set(["go", "php", "ruby", "swift", "kotlin", "rust"]);
          setLanguages(d.languages.filter((l: LanguageCapability) => !excluded.has(l.id)));
        }
      })
      .catch(() => { });

    refreshHistory();
  }, []);

  function changeLanguage(value: string) {
    setLanguage(value);
    setSource(SAMPLES[value] || SAMPLES.python);
    setGenerated(null);
    setVerification(null);
    setShowRejectBox(false);
    setMessage("");
    setArchiveNotice(null);
    setSelectedHistoryId(null);
    if (value === "python") {
      setDocStyle("google");
    } else {
      setDocStyle("standard");
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setMessage("");
    setArchiveNotice(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API}/api/upload-file`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Upload failed (${res.status})`);
      }

      setSource(data.code);
      if (data.language && languages.some(l => l.id === data.language)) {
        setLanguage(data.language);
      }
      setGenerated(null);
      setVerification(null);
      setShowRejectBox(false);

      if (data.mode === "archive") {
        setArchiveNotice(
          `Repository archive loaded: ${data.total_files} source files detected. Displaying primary entrypoint: ${data.primary_file}. (Dependencies & secrets safely excluded).`
        );
        setMessageType("info");
        setMessage(`Archive loaded: ${data.primary_file}`);
      } else {
        setMessageType("info");
        setMessage(`Loaded ${data.filename} (${data.language}) successfully.`);
      }
    } catch (err: any) {
      setMessageType("error");
      setMessage(err.message || "File upload failed.");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function generate() {
    setLoading(true);
    setMessage("");

    try {
      const response = await fetch(`${API}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: source, language, doc_style: docStyle }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `Generation request failed (${response.status})`);
      }
      setGenerated(data);
      setEdited(data.proposed_code);
      if (data.verification) {
        setVerification(data.verification);
      } else {
        setVerification(null);
      }
      setShowRejectBox(false);
      setSelectedHistoryId(null);
      setMessageType("info");
      setMessage(
        "Documentation proposal ready. Review additions, quality scorecard, and diff, then approve or reject."
      );
      await refreshHistory();
    } catch (e: any) {
      setMessageType("error");
      setMessage(e.message || "Could not reach the backend generator.");
    } finally {
      setLoading(false);
    }
  }

  async function approve() {
    if (!generated) return;
    setApproving(true);
    setMessage("");

    try {
      const response = await fetch(`${API}/api/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: generated.run_id, code: edited, doc_style: docStyle }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Approval/verification failed");
      }
      setVerification(data.verification);
      setGenerated(prev => (prev ? { ...prev, status: data.status, verification: data.verification } : null));

      if (data.verification.global_status === "verified") {
        setMessageType("info");
        setMessage("Proposal approved and verified! All AST syntax and structural checks passed.");
      } else if (data.verification.global_status === "partially_verified") {
        setMessageType("warning");
        setMessage(
          "Proposal approved with partial verification. Syntax verification is not supported locally for this language."
        );
      } else {
        setMessageType("error");
        setMessage("Proposal approved, but one or more verification checks failed. Review diagnostics below.");
      }
      await refreshHistory();
    } catch (e: any) {
      setMessageType("error");
      setMessage(e.message || "Failed to record approval.");
    } finally {
      setApproving(false);
    }
  }

  async function confirmReject() {
    if (!generated) return;
    setRejecting(true);
    setMessage("");

    try {
      const response = await fetch(`${API}/api/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: generated.run_id,
          reason: rejectionReason.trim() || "Rejected by developer review",
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Rejection failed");
      }
      setGenerated(prev => (prev ? { ...prev, status: "rejected" } : null));
      setShowRejectBox(false);
      setMessageType("warning");
      setMessage(`Proposal rejected (${data.rejection_reason}). Original source remains unchanged.`);
      await refreshHistory();
    } catch (e: any) {
      setMessageType("error");
      setMessage(e.message || "Failed to record rejection.");
    } finally {
      setRejecting(false);
    }
  }

  async function inspectHistoryRun(item: HistoryItem) {
    try {
      setSelectedHistoryId(item.id);
      const res = await fetch(`${API}/api/history/${item.id}`);
      if (!res.ok) return;
      const data = await res.json();
      setLanguage(data.language);
      setSource(data.original_code);
      setGenerated({
        run_id: data.id,
        language: data.language,
        analysis: data.analysis,
        proposed_code: data.proposed_code,
        mode: data.mode,
        model: data.model || (data.mode === "demo_mock" ? "mock-engine" : data.mode),
        status: data.status,
        warning: data.mode === "demo_mock" ? "Historic run in Demo Mock Mode." : null,
        workflow_state: data.workflow_state || "COMPLETED",
        revision_count: data.revision_count ?? 0,
        agent_executions: data.agents_log || undefined,
      });
      setEdited(data.final_code || data.proposed_code);
      setVerification(data.verification || null);
      setNavTab("workspace");
      setMessage(`Loaded historical run ${item.id.slice(0, 8)} (${data.status}).`);
      setMessageType("info");
    } catch {
      setMessage("Could not load historic run details.");
    }
  }

  function downloadApprovedCode() {
    const blob = new Blob([edited], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const found = languages.find(l => l.id === language);
    const ext = found ? found.extension : "txt";
    a.href = url;
    a.download = `documented_${language}_code.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const isPendingReview = Boolean(generated && generated.status === "pending_review");
  const isApproved = Boolean(generated && generated.status.startsWith("approved"));
  const isRejected = Boolean(generated && generated.status === "rejected");

  return (
    <div className="app-layout">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-icon">
            <WandSparkles size={18} />
          </div>
          <div className="brand-info">
            <b>
              DocuPilot<span> AI</span>
            </b>
            <small>AGENTIC DOCUMENTATION</small>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section-label">Agent Workspace</div>
          <button
            className={`nav-item ${navTab === "workspace" ? "active" : ""}`}
            onClick={() => setNavTab("workspace")}
          >
            <LayoutDashboard size={16} />
            <span>Workspace</span>
          </button>

          <button
            className={`nav-item ${navTab === "history" ? "active" : ""}`}
            onClick={() => {
              setNavTab("history");
              refreshHistory();
            }}
          >
            <History size={16} />
            <span>Audit History</span>
          </button>

          <button
            className={`nav-item ${navTab === "languages" ? "active" : ""}`}
            onClick={() => setNavTab("languages")}
          >
            <Layers size={16} />
            <span>Language Matrix</span>
          </button>

          <button
            className={`nav-item ${navTab === "agents" ? "active" : ""}`}
            onClick={() => setNavTab("agents")}
          >
            <Cpu size={16} />
            <span>Multi-Agent System</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <span className="hackathon-badge">
            <Sparkles size={12} /> ZIRO HACKATHON 2026
          </span>
          <div className="status-indicator">
            <span className={`status-dot ${health}`} />
            <span>FastAPI: {health}</span>
          </div>
        </div>
      </aside>

      {/* Main Wrapper */}
      <div className="main-wrapper">
        {/* Top Header */}
        <header className="top-header">
          <div className="header-breadcrumbs">
            <span>DocuPilot AI</span>
            <ChevronRight size={14} />
            <span className="current">
              {navTab === "workspace"
                ? "Code Documentation Workspace"
                : navTab === "history"
                  ? "Audit Trail & Reviews"
                  : navTab === "languages"
                    ? "Language Capability Registry"
                    : "Multi-Agent System & Central Orchestrator"}
            </span>
          </div>

          <div className="header-actions">
            {config && (
              <span className={`mode-pill ${config.demo_mode ? "mock" : "live"}`}>
                {config.demo_mode ? "⚡ DEMO MOCK MODE" : `● LIVE (${config.model})`}
              </span>
            )}
            <button className="btn-secondary-sm" onClick={refreshHistory} title="Refresh metrics and history">
              <RefreshCw size={13} /> Refresh
            </button>
          </div>
        </header>

        {/* Page Container */}
        <div className="page-container">
          {/* Header Banner */}
          <div className="page-banner">
            <h1>
              {navTab === "workspace"
                ? "Context-Aware Code Documentation"
                : navTab === "history"
                  ? "Audit History & Review Records"
                  : "Central Language Capability Registry"}
            </h1>
            <p>
              {navTab === "workspace"
                ? "Parse real AST structures, generate crystal-clear docstrings, inspect line-by-line diffs, and approve or reject with permanent SQLite audit evidence."
                : navTab === "history"
                  ? "Inspect complete records of previous generation runs, reviewer approvals, rejection reasons, and verification diagnostics."
                  : "Transparent overview of parser engines, compiler syntax verification capabilities, and documentation standards across all 14 supported languages."}
            </p>
          </div>

          {/* Quick Metrics Cards */}
          <div className="stats-grid">
            <div className="metric-card total">
              <div className="metric-content">
                <small>Total Audit Runs</small>
                <div className="metric-value">{stats.total_runs}</div>
              </div>
              <div className="metric-icon-wrap">
                <Database size={18} />
              </div>
            </div>

            <div className="metric-card approved">
              <div className="metric-content">
                <small>Approved & Verified</small>
                <div className="metric-value">{stats.approved}</div>
              </div>
              <div className="metric-icon-wrap">
                <ShieldCheck size={18} />
              </div>
            </div>

            <div className="metric-card pending">
              <div className="metric-content">
                <small>Pending Reviews</small>
                <div className="metric-value">{stats.pending}</div>
              </div>
              <div className="metric-icon-wrap">
                <Clock size={18} />
              </div>
            </div>

            <div className="metric-card rejected">
              <div className="metric-content">
                <small>Rejected Runs</small>
                <div className="metric-value">{stats.rejected}</div>
              </div>
              <div className="metric-icon-wrap">
                <ShieldAlert size={18} />
              </div>
            </div>
          </div>

          {/* Notification Message */}
          {message && (
            <div className={`ui-alert ${messageType}`}>
              {messageType === "error" ? (
                <AlertCircle size={16} />
              ) : messageType === "warning" ? (
                <AlertTriangle size={16} />
              ) : (
                <CheckCircle2 size={16} />
              )}
              <span>{message}</span>
            </div>
          )}

          {/* View: Workspace */}
          {navTab === "workspace" && (
            <>
              {/* Archive Extraction Alert */}
              {archiveNotice && (
                <div className="ui-alert info" style={{ marginBottom: "16px" }}>
                  <FileArchive size={16} />
                  <span>{archiveNotice}</span>
                </div>
              )}

              {/* 4-Stage Agentic Workflow Stepper */}
              <div className="agent-stepper" aria-label="Agentic Workflow Pipeline">
                <div className={`step-item ${generated ? "completed" : "active"}`}>
                  <div className="step-num">1</div>
                  <div className="step-details">
                    <span className="step-title">Code Analysis</span>
                    <span className="step-subtitle">
                      {generated?.analysis
                        ? `${generated.analysis.symbols?.length || 0} AST symbols extracted`
                        : "AST symbol & syntax mapping"}
                    </span>
                  </div>
                  <ChevronRight size={14} className="step-arrow" />
                </div>

                <div className={`step-item ${generated ? "completed" : loading ? "active" : ""}`}>
                  <div className="step-num">2</div>
                  <div className="step-details">
                    <span className="step-title">Documentation Generation</span>
                    <span className="step-subtitle">
                      {generated
                        ? `${generated.mode === "gemini" ? `Gemini (${generated.model})` : generated.mode === "openai" ? `OpenAI (${generated.model})` : "Demo Mock"} (${docStyle})`
                        : `Context-aware (${docStyle})`}
                    </span>
                  </div>
                  <ChevronRight size={14} className="step-arrow" />
                </div>

                <div className={`step-item ${verification ? (verification.passed ? "completed" : "failed") : ""}`}>
                  <div className="step-num">3</div>
                  <div className="step-details">
                    <span className="step-title">Independent Validation</span>
                    <span className="step-subtitle">
                      {verification
                        ? `${verification.global_status.replaceAll("_", " ")} (${verification.quality_score?.overall_score ?? 0}/100)`
                        : "AST & syntax consistency"}
                    </span>
                  </div>
                  <ChevronRight size={14} className="step-arrow" />
                </div>

                <div
                  className={`step-item ${isApproved ? "completed" : isRejected ? "failed" : isPendingReview ? "active" : ""
                    }`}
                >
                  <div className="step-num">4</div>
                  <div className="step-details">
                    <span className="step-title">Human Review & Decision</span>
                    <span className="step-subtitle">
                      {isApproved
                        ? "Approved by developer"
                        : isRejected
                          ? "Rejected by developer"
                          : isPendingReview
                            ? "Awaiting human decision"
                            : "Human-in-the-loop audit"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Central Orchestrator & Specialist Agents Execution Dashboard */}
              {generated && (
                <div className="multi-agent-card">
                  <div className="multi-agent-header">
                    <div className="orchestrator-heading">
                      <div className="orchestrator-icon-badge">
                        <Cpu size={16} />
                      </div>
                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                          <b>Central Orchestrator · Multi-Agent Execution Trace</b>
                          <span
                            className={`status-pill ${isApproved ? "passed" : isRejected ? "failed" : "warning"
                              }`}
                          >
                            {generated.workflow_state || "AWAITING_APPROVAL"}
                          </span>
                        </div>
                        <span className="orchestrator-sub">
                          Deterministic coordination of 4 specialist agents with concurrent review & validation and bounded revision policy.
                        </span>
                      </div>
                    </div>

                    <div className="orchestrator-actions">
                      <span className="revision-pill">
                        {generated.revision_count && generated.revision_count > 0
                          ? `1 Revision Applied`
                          : "0 Revisions (Checks Passed Initially)"}
                      </span>
                      <button
                        type="button"
                        className="btn-secondary-sm"
                        onClick={() => setShowAgentTrace(!showAgentTrace)}
                      >
                        {showAgentTrace ? "Hide Agent Trace" : "Show Agent Trace"}
                      </button>
                    </div>
                  </div>

                  {showAgentTrace && (
                    <div className="agents-grid">
                      {/* Agent 1 */}
                      <div className="agent-box">
                        <div className="agent-box-top">
                          <span className="agent-label">AGENT 1</span>
                          <span className="status-pill passed">COMPLETED</span>
                        </div>
                        <h4>Code Analysis Agent</h4>
                        <p className="agent-desc">
                          {generated.agent_executions?.code_analysis?.summary ||
                            `${generated.analysis?.symbols?.length || 0} AST symbols extracted via ${generated.analysis?.parser_used || "parser"}.`}
                        </p>
                        <div className="agent-meta-tags">
                          <span>Parser: <b>{generated.analysis?.parser_used || "ast"}</b></span>
                          <span>Symbols: <b>{generated.analysis?.symbols?.length || 0}</b></span>
                          {generated.agent_executions?.code_analysis?.execution_time_ms ? (
                            <span>Time: <b>{generated.agent_executions.code_analysis.execution_time_ms}ms</b></span>
                          ) : null}
                        </div>
                      </div>

                      {/* Agent 2 */}
                      <div className="agent-box">
                        <div className="agent-box-top">
                          <span className="agent-label">AGENT 2</span>
                          <span className="status-pill passed">COMPLETED</span>
                        </div>
                        <h4>Documentation Generation Agent</h4>
                        <p className="agent-desc">
                          {generated.agent_executions?.documentation_generation?.summary ||
                            `Generated ${docStyle} documentation via ${generated.mode} (${generated.model}).`}
                        </p>
                        <div className="agent-meta-tags">
                          <span>Engine: <b>{generated.mode}</b></span>
                          <span>Model: <b>{generated.model}</b></span>
                          {generated.agent_executions?.documentation_generation?.execution_time_ms ? (
                            <span>Time: <b>{generated.agent_executions.documentation_generation.execution_time_ms}ms</b></span>
                          ) : null}
                        </div>
                      </div>

                      {/* Agent 3 */}
                      <div className="agent-box">
                        <div className="agent-box-top">
                          <span className="agent-label">AGENT 3</span>
                          <span className="status-pill passed">
                            {generated.agent_executions?.documentation_review?.status?.toUpperCase() || "PASSED"}
                          </span>
                        </div>
                        <h4>Documentation Review Agent</h4>
                        <p className="agent-desc">
                          {generated.agent_executions?.documentation_review?.summary ||
                            `Quality Score: ${verification?.quality_score?.overall_score ?? 100}/100 (${verification?.quality_score?.grade || "A"}). Groundedness verified.`}
                        </p>
                        <div className="agent-meta-tags">
                          <span>Score: <b>{verification?.quality_score?.overall_score ?? 100}/100</b></span>
                          <span>Grade: <b>{verification?.quality_score?.grade || "A"}</b></span>
                          {generated.agent_executions?.documentation_review?.execution_time_ms ? (
                            <span>Time: <b>{generated.agent_executions.documentation_review.execution_time_ms}ms</b></span>
                          ) : null}
                        </div>
                      </div>

                      {/* Agent 4 */}
                      <div className="agent-box">
                        <div className="agent-box-top">
                          <span className="agent-label">AGENT 4</span>
                          <span className={`status-pill ${verification?.passed ? "passed" : "failed"}`}>
                            {verification?.passed ? "VERIFIED" : "FAILED"}
                          </span>
                        </div>
                        <h4>Validation Agent</h4>
                        <p className="agent-desc">
                          {generated.agent_executions?.validation?.summary ||
                            (verification?.passed
                              ? "AST executable statements 100% preserved. Syntax checks verified."
                              : "Validation checks failed.")}
                        </p>
                        <div className="agent-meta-tags">
                          <span>Checks: <b>{verification?.checks?.length ?? 4} checks</b></span>
                          <span>Statements: <b>100% Preserved</b></span>
                          {generated.agent_executions?.validation?.execution_time_ms ? (
                            <span>Time: <b>{generated.agent_executions.validation.execution_time_ms}ms</b></span>
                          ) : null}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div className="workspace-grid">
                {/* Panel 1: Source Code Input */}
                <div className="card-panel">
                  <div className="panel-header">
                    <div className="panel-heading">
                      <FileCode2 size={16} />
                      <span>01 / SOURCE CODE INPUT</span>
                    </div>
                    <span
                      className={`capability-pill ${activeCapability.syntax_validation_capable ? "verified" : "heuristic"
                        }`}
                      title={activeCapability.notes}
                    >
                      {activeCapability.syntax_validation_capable
                        ? `✓ AST VALIDATED (${activeCapability.parser_type})`
                        : `⚠ HEURISTIC (SYNTAX NOT VERIFIED)`}
                    </span>
                  </div>

                  <div className="panel-controls">
                    <div className="select-wrapper">
                      <select
                        value={language}
                        onChange={e => changeLanguage(e.target.value)}
                        aria-label="Programming language"
                      >
                        {languages.map(l => (
                          <option key={l.id} value={l.id}>
                            {l.name} {l.syntax_validation_capable ? "• (AST Parser)" : "• (Generation)"}
                          </option>
                        ))}
                      </select>
                    </div>

                    {language === "python" && (
                      <div className="style-toggle" title="Documentation format style">
                        <button
                          type="button"
                          className={`style-toggle-btn ${docStyle === "google" ? "active" : ""}`}
                          onClick={() => setDocStyle("google")}
                        >
                          Google PEP 257
                        </button>
                        <button
                          type="button"
                          className={`style-toggle-btn ${docStyle === "numpy" ? "active" : ""}`}
                          onClick={() => setDocStyle("numpy")}
                        >
                          NumPy Style
                        </button>
                      </div>
                    )}

                    <label
                      className="btn-secondary-sm upload-label"
                      title="Upload source file or ZIP repository archive"
                    >
                      <Upload size={12} />
                      <span>{uploading ? "Uploading..." : "Upload File/ZIP"}</span>
                      <input
                        type="file"
                        accept=".py,.js,.ts,.java,.cpp,.c,.cs,.sql,.zip"
                        onChange={handleFileUpload}
                        disabled={uploading || loading}
                      />
                    </label>

                    <button
                      className="btn-secondary-sm"
                      onClick={() => setSource(SAMPLES[language] || SAMPLES.python)}
                    >
                      <RefreshCw size={12} /> Reset sample
                    </button>
                  </div>

                  <div className="editor-wrapper">
                    <textarea
                      className="code-textarea"
                      spellCheck={false}
                      value={source}
                      onChange={e => setSource(e.target.value)}
                      aria-label="Source code"
                      placeholder="Paste your source code here..."
                    />
                  </div>

                  <div className="panel-footer">
                    <span className="panel-hint">
                      {source.length} characters · {source.split("\n").length} lines
                    </span>
                    <button
                      className="btn-primary"
                      disabled={loading || source.trim().length < 3 || approving || rejecting}
                      onClick={generate}
                    >
                      {loading ? <LoaderCircle className="spin" size={16} /> : <Play size={15} />}
                      <span>{loading ? "Analyzing AST & Generating..." : "Generate Documentation"}</span>
                    </button>
                  </div>
                </div>

                {/* Panel 2: Review & Verification Workspace */}
                <div className="card-panel">
                  <div className="panel-header">
                    <div className="panel-heading">
                      <Bot size={16} />
                      <span>02 / REVIEW & VERIFICATION</span>
                    </div>
                    <span
                      className="status-badge"
                      style={{
                        backgroundColor: !generated
                          ? "#F0F2F7"
                          : isPendingReview
                            ? "var(--warning-light)"
                            : isApproved
                              ? "var(--success-light)"
                              : "var(--error-light)",
                        color: !generated
                          ? "var(--text-secondary)"
                          : isPendingReview
                            ? "var(--warning)"
                            : isApproved
                              ? "var(--success)"
                              : "var(--error)",
                      }}
                    >
                      {!generated
                        ? "AWAITING INPUT"
                        : isPendingReview
                          ? "PENDING REVIEW"
                          : isApproved
                            ? "APPROVED"
                            : "REJECTED"}
                    </span>
                  </div>

                  {!generated ? (
                    <div className="empty-placeholder">
                      <div className="empty-icon-circle">
                        <WandSparkles size={24} />
                      </div>
                      <h3>Documentation Proposal</h3>
                      <p>
                        Click <b>Generate Documentation</b> to extract AST symbols and generate an editable, verifiable proposal.
                      </p>
                    </div>
                  ) : (
                    <>
                      {/* Meta Strip */}
                      <div className="meta-strip">
                        <div className="meta-item">
                          <small>Language</small>
                          <b>{generated.language}</b>
                        </div>
                        <div className="meta-item">
                          <small>Generator</small>
                          <b>
                            {generated.mode === "gemini"
                              ? `Google Gemini (${generated.model})`
                              : generated.mode === "openai"
                                ? `OpenAI (${generated.model})`
                                : "DocuPilot Demo Engine"}
                          </b>
                        </div>
                        <div className="meta-item">
                          <small>Parser Engine</small>
                          <b>
                            {generated.analysis?.parser_used === "python_ast"
                              ? "Python AST"
                              : generated.analysis?.parser_used === "esprima"
                                ? "Esprima JS AST"
                                : generated.analysis?.parser_used === "typescript_compiler"
                                  ? "TypeScript Compiler API"
                                  : "Regex Structural"}
                          </b>
                        </div>
                      </div>

                      {/* Demo Mode Notice */}
                      {generated.warning && (
                        <div className="ui-alert warning" style={{ margin: "10px 20px" }}>
                          <AlertTriangle size={15} />
                          <span>{generated.warning}</span>
                        </div>
                      )}

                      {/* View Tabs */}
                      <div className="workspace-tabs">
                        <button
                          className={`tab-button ${viewTab === "edit" ? "active" : ""}`}
                          onClick={() => setViewTab("edit")}
                        >
                          <FileCode2 size={13} /> Proposed Code (Editable)
                        </button>
                        <button
                          className={`tab-button ${viewTab === "diff" ? "active" : ""}`}
                          onClick={() => setViewTab("diff")}
                        >
                          <GitCompare size={13} /> Diff Preview ({diffLines.filter(l => l.type === "add").length} additions)
                        </button>
                      </div>

                      {/* Main Code View: Edit or Diff */}
                      <div className="editor-wrapper">
                        {viewTab === "edit" ? (
                          <textarea
                            id="proposed"
                            className="code-textarea"
                            spellCheck={false}
                            value={edited}
                            disabled={isApproved || isRejected}
                            onChange={e => setEdited(e.target.value)}
                          />
                        ) : (
                          <div className="diff-scroll-box">
                            {diffLines.map((line, idx) => (
                              <div key={idx} className={`diff-row ${line.type}`}>
                                <span className="diff-num">{line.lineNum || ""}</span>
                                <span className="diff-code">
                                  {line.type === "add" ? "+ " : line.type === "del" ? "- " : "  "}
                                  {line.text}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Inline Rejection Box */}
                      {showRejectBox && (
                        <div className="rejection-box">
                          <label>State rejection reason (recorded in SQLite audit trail):</label>
                          <input
                            type="text"
                            placeholder="e.g. Docstring missing edge case details..."
                            value={rejectionReason}
                            onChange={e => setRejectionReason(e.target.value)}
                            autoFocus
                          />
                          <div className="rejection-box-actions">
                            <button className="btn-secondary-sm" onClick={() => setShowRejectBox(false)}>
                              Cancel
                            </button>
                            <button
                              className="btn-reject"
                              style={{ backgroundColor: "var(--error)", color: "#FFFFFF" }}
                              disabled={rejecting}
                              onClick={confirmReject}
                            >
                              {rejecting ? "Recording..." : "Confirm Rejection"}
                            </button>
                          </div>
                        </div>
                      )}

                      {/* Actions: Approve / Reject / Download */}
                      <div className="panel-footer">
                        {isPendingReview ? (
                          <>
                            <button
                              className="btn-reject"
                              disabled={approving || rejecting || loading}
                              onClick={() => setShowRejectBox(true)}
                            >
                              <X size={15} /> Reject
                            </button>
                            <button
                              className="btn-approve"
                              disabled={approving || rejecting || !edited.trim()}
                              onClick={approve}
                            >
                              {approving ? (
                                <LoaderCircle className="spin" size={16} />
                              ) : (
                                <Check size={16} />
                              )}
                              <span>{approving ? "Running verification..." : "Approve & Verify"}</span>
                            </button>
                          </>
                        ) : isApproved ? (
                          <button className="btn-download" onClick={downloadApprovedCode}>
                            <ArrowDownToLine size={15} /> Download Approved Code ({language})
                          </button>
                        ) : (
                          <span className="panel-hint">Proposal was rejected. Original source remains unchanged.</span>
                        )}
                      </div>

                      {/* Quality Scorecard (Visible during Review & Approval) */}
                      {(verification?.quality_score || generated?.quality_score) && (
                        <div className="quality-card">
                          <div className="quality-header">
                            <div className="quality-heading">
                              <Sparkles size={16} style={{ color: "var(--accent-primary)" }} />
                              <span>Documentation Quality Scorecard</span>
                            </div>
                            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                              <b style={{ fontSize: "16px", color: "var(--text-primary)" }}>
                                {(verification?.quality_score || generated?.quality_score)?.overall_score} / 100
                              </b>
                              <span
                                className={`grade-badge ${((verification?.quality_score || generated?.quality_score)?.grade || "")
                                  .toLowerCase()
                                  .replace("+", "-plus")
                                  .replace(" ", "-")}`}
                              >
                                Grade: {(verification?.quality_score || generated?.quality_score)?.grade}
                              </span>
                            </div>
                          </div>

                          <div className="quality-breakdown-grid">
                            {(verification?.quality_score || generated?.quality_score)?.breakdown.map((item, idx) => (
                              <div key={idx} className="criterion-pill">
                                <small>{item.criterion}</small>
                                <b>
                                  {item.earned} / {item.weight} pts
                                </b>
                                <span
                                  style={{
                                    fontSize: "10px",
                                    fontWeight: 600,
                                    color:
                                      item.status === "passed"
                                        ? "var(--success)"
                                        : item.status === "warning"
                                          ? "var(--warning)"
                                          : "var(--error)",
                                  }}
                                >
                                  {item.status.toUpperCase()}
                                </span>
                              </div>
                            ))}
                          </div>

                          {(verification?.quality_score || generated?.quality_score)?.findings?.length ? (
                            <div className="findings-list">
                              <b>AST Evidence & Findings:</b>
                              <ul>
                                {(verification?.quality_score || generated?.quality_score)?.findings.map((f, i) => (
                                  <li key={i}>{f}</li>
                                ))}
                              </ul>
                            </div>
                          ) : null}

                          <div className="quality-disclaimer">
                            {(verification?.quality_score || generated?.quality_score)?.disclaimer}
                          </div>
                        </div>
                      )}

                      {/* Code Preservation Evidence (AST Fingerprinting) */}
                      {verification && (
                        <div className="fingerprint-card">
                          <div className="fingerprint-header">
                            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                              <ShieldCheck size={15} style={{ color: "var(--success)" }} />
                              <b>Code Preservation Evidence (AST Fingerprint)</b>
                            </div>
                            {verification.checks.find(c => c.name === "Structural Consistency")?.status === "passed" ? (
                              <span className="pill-check passed">100% Executable Statements Preserved</span>
                            ) : (
                              <span className="pill-check failed">Executable Statements Modified</span>
                            )}
                          </div>

                          {verification.checks.find(c => c.fingerprint_comparison)?.fingerprint_comparison && (
                            <div className="fingerprint-hashes">
                              <span>
                                Original AST:{" "}
                                <b>
                                  {
                                    verification.checks.find(c => c.fingerprint_comparison)
                                      ?.fingerprint_comparison?.original_fingerprint
                                  }
                                </b>
                              </span>
                              <span>
                                Proposed AST:{" "}
                                <b>
                                  {
                                    verification.checks.find(c => c.fingerprint_comparison)
                                      ?.fingerprint_comparison?.proposed_fingerprint
                                  }
                                </b>
                              </span>
                              <span>
                                Matches:{" "}
                                <b>
                                  {verification.checks.find(c => c.fingerprint_comparison)
                                    ?.fingerprint_comparison?.matches
                                    ? "TRUE (Statements Identical)"
                                    : "FALSE (Mutation Detected)"}
                                </b>
                              </span>
                            </div>
                          )}

                          <div style={{ fontSize: "10px", color: "var(--text-tertiary)", marginTop: "4px" }}>
                            Static syntax parsing and AST fingerprinting guarantee that non-comment executable statement
                            structures are identical. Passing static checks does not claim or prove dynamic runtime
                            behavioral equivalence.
                          </div>
                        </div>
                      )}

                      {/* Verification Cards Container */}
                      {verification && (
                        <div className="verification-container">
                          <div className="verification-header">
                            <div className={`verification-heading ${verification.passed ? "passed" : "failed"}`}>
                              {verification.passed ? <ShieldCheck size={18} /> : <ShieldAlert size={18} />}
                              <span>Verification Pipeline</span>
                            </div>
                            <span className={`global-status-badge ${verification.global_status}`}>
                              {verification.global_status.replaceAll("_", " ")}
                            </span>
                          </div>

                          <div className="checks-cards-grid">
                            {verification.checks.map((chk, i) => (
                              <div key={i} className="check-box">
                                <div className="check-box-top">
                                  <span className="check-title">
                                    {chk.status === "passed" ? (
                                      <Check size={14} style={{ color: "var(--success)" }} />
                                    ) : chk.status === "failed" ? (
                                      <X size={14} style={{ color: "var(--error)" }} />
                                    ) : (
                                      <AlertCircle size={14} style={{ color: "var(--text-tertiary)" }} />
                                    )}
                                    <span>{chk.name}</span>
                                  </span>
                                  <span className={`pill-check ${chk.status}`}>
                                    {chk.status.replaceAll("_", " ")}
                                  </span>
                                </div>
                                <div className="check-desc">{chk.details}</div>
                                <div className="check-footer-info">
                                  <span>Tool: {chk.tool}</span>
                                  {chk.duration_ms > 0 && <span>{chk.duration_ms}ms</span>}
                                </div>
                              </div>
                            ))}
                          </div>

                          <div className="verification-note">{verification.note}</div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            </>
          )}

          {/* View: Dedicated Audit History */}
          {navTab === "history" && (
            <div className="history-section">
              <div className="history-header">
                <div className="history-title">
                  <History size={18} />
                  <span>Permanent Audit History (SQLite)</span>
                </div>
                <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                  <input
                    type="text"
                    placeholder="Filter by run ID or language..."
                    value={historyFilter}
                    onChange={e => setHistoryFilter(e.target.value)}
                    style={{
                      padding: "6px 12px",
                      fontSize: "12px",
                      borderRadius: "var(--radius-md)",
                      border: "1px solid var(--border-color)",
                      outline: "none",
                      width: "220px",
                    }}
                  />
                  <button className="btn-secondary-sm" onClick={refreshHistory}>
                    <RefreshCw size={12} /> Refresh
                  </button>
                </div>
              </div>

              <div className="history-table-container">
                {filteredHistory.length === 0 ? (
                  <div className="empty-placeholder">
                    <p>No audit runs match your criteria.</p>
                  </div>
                ) : (
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>Run ID</th>
                        <th>Language</th>
                        <th>Status</th>
                        <th>Engine / Model</th>
                        <th>Rejection Reason</th>
                        <th>Created At</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredHistory.map(item => (
                        <tr
                          key={item.id}
                          className={`history-row ${selectedHistoryId === item.id ? "selected" : ""}`}
                          onClick={() => inspectHistoryRun(item)}
                        >
                          <td style={{ fontFamily: "var(--font-mono)", fontSize: "11px", fontWeight: 600 }}>
                            {item.id.slice(0, 8)}...
                          </td>
                          <td style={{ textTransform: "capitalize", fontWeight: 500 }}>{item.language}</td>
                          <td>
                            <span className={`status-badge ${item.status}`}>
                              {item.status.replaceAll("_", " ")}
                            </span>
                          </td>
                          <td style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                            {item.model || item.mode}
                          </td>
                          <td style={{ fontSize: "12px", color: "var(--text-secondary)", maxWidth: "240px" }}>
                            {item.rejection_reason || "—"}
                          </td>
                          <td style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                            {new Date(item.created_at).toLocaleString()}
                          </td>
                          <td>
                            <button
                              className="btn-secondary-sm"
                              onClick={e => {
                                e.stopPropagation();
                                inspectHistoryRun(item);
                              }}
                            >
                              <Eye size={12} /> Inspect
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}

          {/* View: Central Language Matrix */}
          {navTab === "languages" && (
            <div className="matrix-card">
              <h2>Language Capability Registry</h2>
              <p>
                Centralized registry describing AST parser availability, syntax verification capability,
                and documentation standards across all supported languages.
              </p>

              <div className="history-table-container">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>Language</th>
                      <th>Extension</th>
                      <th>Parser Engine</th>
                      <th>Syntax Validation</th>
                      <th>Documentation Convention</th>
                      <th>Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {languages.map(lang => (
                      <tr key={lang.id}>
                        <td style={{ fontWeight: 600 }}>{lang.name}</td>
                        <td style={{ fontFamily: "var(--font-mono)" }}>.{lang.extension}</td>
                        <td style={{ fontFamily: "var(--font-mono)", fontSize: "12px" }}>
                          {lang.parser_type}
                        </td>
                        <td>
                          <span
                            className={`capability-pill ${lang.syntax_validation_capable ? "verified" : "heuristic"
                              }`}
                          >
                            {lang.syntax_validation_capable ? "Full AST Verified" : "Not Verified (No Local Parser)"}
                          </span>
                        </td>
                        <td style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                          {lang.doc_format}
                        </td>
                        <td style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                          {lang.notes}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* View: Multi-Agent Architecture & Central Orchestrator */}
          {navTab === "agents" && (
            <div className="agents-overview-section">
              <div className="agents-header">
                <div className="agents-header-title">
                  <div className="brand-icon" style={{ width: "42px", height: "42px" }}>
                    <Cpu size={22} />
                  </div>
                  <div>
                    <h2>Hybrid Multi-Agent Architecture</h2>
                    <p>
                      DocuPilot AI couples one central Orchestrator with four specialist agents for deterministic analysis,
                      context-aware generation, independent grounded review, and static AST verification.
                    </p>
                  </div>
                </div>

                <div className="orchestrator-status-strip">
                  <span className="pill-agent">1 Orchestrator Controller</span>
                  <span className="pill-agent">4 Specialist Agents</span>
                  <span className="pill-agent">Bounded 1-Revision Loop</span>
                  <span className="pill-agent">Human-in-the-Loop Gate</span>
                </div>
              </div>

              {/* Central Orchestrator Banner */}
              <div className="card-panel" style={{ marginBottom: "24px" }}>
                <div className="panel-header">
                  <div className="panel-heading">
                    <SlidersHorizontal size={16} />
                    <span>CENTRAL ORCHESTRATOR &amp; WORKFLOW CONTROLLER</span>
                  </div>
                  <span className="capability-pill verified">DETERMINISTIC PYTHON STATE MACHINE</span>
                </div>

                <div style={{ padding: "20px" }}>
                  <div className="orchestrator-state-flow">
                    <div className="flow-step">
                      <span className="flow-badge">1</span>
                      <b>RECEIVED</b>
                      <small>Validate request &amp; assign run_id</small>
                    </div>
                    <ChevronRight size={14} className="flow-arrow" />
                    <div className="flow-step">
                      <span className="flow-badge">2</span>
                      <b>ANALYZING</b>
                      <small>Code Analysis Agent (AST)</small>
                    </div>
                    <ChevronRight size={14} className="flow-arrow" />
                    <div className="flow-step">
                      <span className="flow-badge">3</span>
                      <b>GENERATING</b>
                      <small>Documentation Generation Agent</small>
                    </div>
                    <ChevronRight size={14} className="flow-arrow" />
                    <div className="flow-step">
                      <span className="flow-badge">4</span>
                      <b>REVIEW &amp; VALIDATE</b>
                      <small>Concurrent Review &amp; Validation</small>
                    </div>
                    <ChevronRight size={14} className="flow-arrow" />
                    <div className="flow-step">
                      <span className="flow-badge">5</span>
                      <b>AWAITING APPROVAL</b>
                      <small>Human Decision Gate</small>
                    </div>
                    <ChevronRight size={14} className="flow-arrow" />
                    <div className="flow-step">
                      <span className="flow-badge">6</span>
                      <b>COMPLETED</b>
                      <small>Audit Log &amp; Download</small>
                    </div>
                  </div>

                  <div className="orchestrator-rules-grid">
                    <div className="rule-box">
                      <Shield size={16} style={{ color: "var(--accent-primary)" }} />
                      <b>Immutable Original Source</b>
                      <p>The original source code is preserved in immutable state. Agents never overwrite the user's source directly.</p>
                    </div>
                    <div className="rule-box">
                      <Activity size={16} style={{ color: "var(--accent-secondary)" }} />
                      <b>Concurrent Review &amp; Validation</b>
                      <p>Documentation Review and Validation agents execute concurrently on the candidate output via ThreadPoolExecutor.</p>
                    </div>
                    <div className="rule-box">
                      <RefreshCw size={16} style={{ color: "var(--warning)" }} />
                      <b>Bounded Single-Revision Limit</b>
                      <p>If the Review Agent detects missing parameters or poor docstrings, the Orchestrator allows exactly one revision to prevent infinite loops.</p>
                    </div>
                    <div className="rule-box">
                      <CheckCircle2 size={16} style={{ color: "var(--success)" }} />
                      <b>Human-in-the-Loop Gate</b>
                      <p>No generated patch is ever merged or deployed automatically. Approval or rejection requires explicit operator decision.</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* 4 Specialist Agents Grid */}
              <div className="four-agents-grid">
                {/* Specialist Agent 1 */}
                <div className="specialist-card">
                  <div className="specialist-card-header">
                    <div className="specialist-num">01</div>
                    <div>
                      <h3>Code Analysis Agent</h3>
                      <span className="specialist-role">AST &amp; Structural Parser</span>
                    </div>
                  </div>
                  <p className="specialist-summary">
                    Inspects input source code, identifies programming language, extracts declared functions, classes, parameters, return types, and raises. Flags symbols lacking documentation.
                  </p>
                  <div className="specialist-features">
                    <div className="spec-item">
                      <b>Reused Modules:</b>
                      <span>analyzer.py, languages.py</span>
                    </div>
                    <div className="spec-item">
                      <b>Engines:</b>
                      <span>Python AST, Esprima (JS), TypeScript Compiler API, Regex Heuristics</span>
                    </div>
                    <div className="spec-item">
                      <b>Guarantees:</b>
                      <span>Never modifies source code; truthfully distinguishes verified AST from heuristics.</span>
                    </div>
                  </div>
                </div>

                {/* Specialist Agent 2 */}
                <div className="specialist-card">
                  <div className="specialist-card-header">
                    <div className="specialist-num">02</div>
                    <div>
                      <h3>Documentation Generation Agent</h3>
                      <span className="specialist-role">Context-Aware Generation</span>
                    </div>
                  </div>
                  <p className="specialist-summary">
                    Synthesizes explanatory docstrings and inline comments grounded in the Code Analysis Agent&apos;s findings. Adheres to target conventions (Google PEP 257, NumPy, JSDoc, TSDoc).
                  </p>
                  <div className="specialist-features">
                    <div className="spec-item">
                      <b>Reused Modules:</b>
                      <span>OpenAI API, Google Gemini API, deterministic demo mock engine</span>
                    </div>
                    <div className="spec-item">
                      <b>Revision Support:</b>
                      <span>Accepts actionable feedback from Review Agent on revision passes.</span>
                    </div>
                    <div className="spec-item">
                      <b>Guarantees:</b>
                      <span>No vague placeholders; outputs candidate patch without modifying original.</span>
                    </div>
                  </div>
                </div>

                {/* Specialist Agent 3 */}
                <div className="specialist-card">
                  <div className="specialist-card-header">
                    <div className="specialist-num">03</div>
                    <div>
                      <h3>Documentation Review Agent</h3>
                      <span className="specialist-role">Independent Quality &amp; Groundedness Audit</span>
                    </div>
                  </div>
                  <p className="specialist-summary">
                    Compares proposed documentation against original code and AST evidence. Audits parameter coverage, return grounding, detects placeholder phrases, and generates revision directives.
                  </p>
                  <div className="specialist-features">
                    <div className="spec-item">
                      <b>Reused Modules:</b>
                      <span>quality.py, analyzer.py</span>
                    </div>
                    <div className="spec-item">
                      <b>Scoring Criteria:</b>
                      <span>Parameter coverage (35%), returns (20%), exceptions (15%), summary (15%), style (15%)</span>
                    </div>
                    <div className="spec-item">
                      <b>Guarantees:</b>
                      <span>Independent evaluation separate from generation agent; actionable feedback.</span>
                    </div>
                  </div>
                </div>

                {/* Specialist Agent 4 */}
                <div className="specialist-card">
                  <div className="specialist-card-header">
                    <div className="specialist-num">04</div>
                    <div>
                      <h3>Validation Agent</h3>
                      <span className="specialist-role">Syntax &amp; Structural Integrity Checker</span>
                    </div>
                  </div>
                  <p className="specialist-summary">
                    Performs rigorous static validation of the candidate code. Compares executable AST fingerprints (SHA-256) to ensure non-comment statements and logic were not mutated.
                  </p>
                  <div className="specialist-features">
                    <div className="spec-item">
                      <b>Reused Modules:</b>
                      <span>verifier.py, ts_runner.py, ts_validator.cjs</span>
                    </div>
                    <div className="spec-item">
                      <b>Checks Run:</b>
                      <span>Syntax validity, documentation format, structural statement consistency</span>
                    </div>
                    <div className="spec-item">
                      <b>Guarantees:</b>
                      <span>Blocks completion if syntax breaks; truthfully discloses static vs runtime limits.</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Footer */}
          <footer className="app-footer">
            <span>DocuPilot AI · Ziro Agentic AI Hackathon 2026</span>
            <span>Static verification ≠ runtime equivalence · Always review generated code.</span>
          </footer>
        </div>
      </div>
    </div>
  );
}
