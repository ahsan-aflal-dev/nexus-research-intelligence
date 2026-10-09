
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  FileCheck2,
  FileText,
  FolderOpen,
  Gauge,
  LoaderCircle,
  Menu,
  MessageSquareText,
  Network,
  Plus,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Trash2,
  Upload,
  X,
  Zap,
} from "lucide-react";
import "./index.css";

type Source = {
  citation?: string;
  source?: string;
  page_number?: number | null;
  chunk_id?: string;
  similarity?: number;
  text?: string;
};

type DocumentItem = {
  source_path?: string;
  file_path?: string;
  filename?: string;
  name?: string;
  source?: string;
  page_count?: number;
  chunk_count?: number;
  status?: string;
  [key: string]: unknown;
};

type ApiHealth = {
  status?: string;
  service?: string;
};

type EvaluationMetrics = {
  question_count: number;
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  mrr: number;
  questions_with_missing_labels: number;
  questions_with_no_relevant_result: number;
};

type RetrievedEvaluationResult = {
  rank: number;
  chunk_id: string;
  source: string;
  page_number: number;
  similarity: number;
  is_relevant: boolean;
};

type EvaluationQuestion = {
  id: string;
  question: string;
  first_relevant_rank: number | null;
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  reciprocal_rank: number;
  missing_relevant_chunk_ids?: string[];
  retrieved_results?: RetrievedEvaluationResult[];
};

type EvaluationReport = {
  dataset_name: string;
  dataset_version?: string;
  evaluated_at_utc?: string;
  index_directory?: string;
  embedding_model?: string;
  indexed_chunk_count: number;
  top_k: number;
  metrics: EvaluationMetrics;
  questions?: EvaluationQuestion[];
};

type EvaluationResponse = {
  available?: boolean;
  status?: string;
  message?: string;
  report?: EvaluationReport;
};

type Answer = {
  question: string;
  answer: string;
  sources: Source[];
  source_count: number;
};

type Section = "Ask NEXUS" | "Knowledge Base" | "System Monitor";

const NAV_ITEMS: {
  label: Section;
  description: string;
  icon: typeof MessageSquareText;
}[] = [
  {
    label: "Ask NEXUS",
    description: "Research workspace",
    icon: MessageSquareText,
  },
  {
    label: "Knowledge Base",
    description: "Your documents",
    icon: BookOpen,
  },
  {
    label: "System Monitor",
    description: "System health",
    icon: Activity,
  },
];

const SUGGESTIONS = [
  {
    title: "Summarize my research",
    description: "Find the main ideas across my documents.",
    question: "Summarize the main ideas in my documents.",
    icon: BookOpen,
  },
  {
    title: "Find supporting evidence",
    description: "Locate passages that support key conclusions.",
    question: "What evidence supports the key conclusions?",
    icon: Search,
  },
  {
    title: "Compare sources",
    description: "Identify shared and contrasting concepts.",
    question: "Compare the main concepts across my sources.",
    icon: Network,
  },
];

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);

  if (!response.ok) {
    let message = `Request failed (${response.status})`;

    try {
      const data = await response.json();
      message = data.detail || data.message || message;
    } catch {
      // Keep the HTTP status when the server returns non-JSON content.
    }

    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

function displayName(document: DocumentItem): string {
  const path =
    document.filename ||
    document.name ||
    document.file_path ||
    document.source_path ||
    document.source ||
    "Indexed document";

  return String(path).split(/[\\/]/).pop() || String(path);
}

function formatDate(value?: string): string {
  if (!value) return "Not available";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return "Not available";

  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function LogoMark({ size = 40 }: { size?: number }) {
  const gradientId = `nexus-logo-${size}`;

  return (
    <svg
      className="nexus-logo"
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      role="img"
      aria-label="NEXUS logo"
    >
      <rect
        x="1"
        y="1"
        width="46"
        height="46"
        rx="15"
        fill={`url(#${gradientId}-bg)`}
        stroke="rgba(190,180,255,.3)"
      />
      <path
        d="M13 32V16L35 32V16"
        stroke={`url(#${gradientId}-line)`}
        strokeWidth="3.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="13" cy="16" r="2.4" fill="#E2DEFF" />
      <circle cx="35" cy="32" r="2.4" fill="#A99AFF" />
      <defs>
        <linearGradient
          id={`${gradientId}-bg`}
          x1="5"
          y1="4"
          x2="43"
          y2="44"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#292345" />
          <stop offset="1" stopColor="#141B2A" />
        </linearGradient>
        <linearGradient
          id={`${gradientId}-line`}
          x1="13"
          y1="16"
          x2="35"
          y2="32"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#F0EDFF" />
          <stop offset="1" stopColor="#9B8CFF" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function App() {
  const [section, setSection] = useState<Section>("Ask NEXUS");
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const [health, setHealth] = useState<ApiHealth | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);

  const [question, setQuestion] = useState("");
  const [documentSearch, setDocumentSearch] = useState("");
  const [topK, setTopK] = useState(5);
  const [answer, setAnswer] = useState<Answer | null>(null);

  const [asking, setAsking] = useState(false);
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [refreshingStatus, setRefreshingStatus] = useState(false);
  const [deletingDocument, setDeletingDocument] = useState<string | null>(
    null,
  );

  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const [evaluation, setEvaluation] =
    useState<EvaluationResponse | null>(null);
  const [loadingEvaluation, setLoadingEvaluation] = useState(false);
  const [runningEvaluation, setRunningEvaluation] = useState(false);
  const [evaluationQuestionsOpen, setEvaluationQuestionsOpen] =
    useState(true);

  const fileInput = useRef<HTMLInputElement>(null);
  const questionInput = useRef<HTMLTextAreaElement>(null);

  const refreshHealth = useCallback(async () => {
    try {
      const result = await api<ApiHealth>("/health");
      setHealth(result);
    } catch {
      setHealth(null);
    }
  }, []);

  const refreshDocuments = useCallback(async () => {
    setLoadingDocuments(true);

    try {
      const result = await api<{ documents?: DocumentItem[] }>("/documents");
      setDocuments(Array.isArray(result.documents) ? result.documents : []);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not load documents.",
      );
    } finally {
      setLoadingDocuments(false);
    }
  }, []);

  const refreshEvaluation = useCallback(async () => {
    setLoadingEvaluation(true);

    try {
      setEvaluation(await api<EvaluationResponse>("/evaluation/latest"));
    } catch {
      setEvaluation({
        available: false,
        message:
          "Could not load the evaluation report. Check the API connection.",
      });
    } finally {
      setLoadingEvaluation(false);
    }
  }, []);

  // Load the initial dashboard data and periodically check API availability.
  useEffect(() => {
    const initialTimer = window.setTimeout(() => {
      void refreshHealth();
      void refreshDocuments();
      void refreshEvaluation();
    }, 0);

    const healthTimer = window.setInterval(() => {
      void refreshHealth();
    }, 15_000);

    return () => {
      window.clearTimeout(initialTimer);
      window.clearInterval(healthTimer);
    };
  }, [refreshHealth, refreshDocuments, refreshEvaluation]);

  // Escape closes the mobile navigation panel.
  useEffect(() => {
    if (!mobileNavOpen) return;

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setMobileNavOpen(false);
      }
    }

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [mobileNavOpen]);

  function navigate(nextSection: Section) {
    setSection(nextSection);
    setMobileNavOpen(false);
    setError("");
    setNotice("");
  }

  async function askQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmed = question.trim();
    if (!trimmed || asking) return;

    setAsking(true);
    setError("");
    setNotice("");
    setAnswer(null);

    try {
      const result = await api<Answer>("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, top_k: topK }),
      });

      setAnswer(result);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to query NEXUS. Check that the API is running.",
      );
    } finally {
      setAsking(false);
    }
  }

  async function uploadPdf(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please choose a PDF file.");
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      setError("PDFs must be 25 MB or smaller.");
      return;
    }

    setUploading(true);
    setError("");
    setNotice("");

    try {
      const form = new FormData();
      form.append("file", file);

      const result = await api<{
        message?: string;
        status?: string;
        filename?: string;
      }>("/documents/upload", {
        method: "POST",
        body: form,
      });

      setNotice(
        result.message ||
          `Upload processed${result.filename ? `: ${result.filename}` : "."}`,
      );

      await Promise.all([refreshDocuments(), refreshHealth()]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Upload failed. Check the backend upload endpoint.",
      );
    } finally {
      setUploading(false);
    }
  }


  async function deleteDocument(document: DocumentItem) {
    const filePath = document.source_path;

    if (!filePath) {
      setError(
        "This document has no source path, so NEXUS cannot safely remove it.",
      );
      setNotice("");
      return;
    }

    const name = displayName(document);
    const confirmed = window.confirm(
      `Remove "${name}" from the NEXUS index? The backend determines whether the original PDF is also deleted.`,
    );

    if (!confirmed) return;

    setDeletingDocument(filePath);
    setError("");
    setNotice("");

    try {
      const result = await api<{
        message?: string;
        status?: string;
        file_deleted?: boolean;
      }>("/documents", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: filePath }),
      });

      // Remove the document from the visible dashboard after DELETE succeeds.
      setDocuments((current) =>
        current.filter((item) => item.source_path !== filePath),
      );
      setAnswer(null);

      // Refresh from the backend to confirm the actual registry.
      try {
        const refreshed = await api<{ documents?: DocumentItem[] }>(
          "/documents",
        );

        setDocuments(
          Array.isArray(refreshed.documents) ? refreshed.documents : [],
        );
        setNotice(result.message || `"${name}" was removed from NEXUS.`);
        setError("");
      } catch {
        // The delete request succeeded, but the registry could not be verified.
        setNotice("");
        setError(
          `"${name}" was removed from the current dashboard, but NEXUS could not verify the latest document registry. Use Refresh documents to retry.`,
        );
      }

      await refreshHealth();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not delete the document. Check the backend terminal.",
      );
      setNotice("");
    } finally {
      setDeletingDocument(null);
    }
  }


  async function runEvaluation() {
    if (runningEvaluation) return;

    setRunningEvaluation(true);
    setError("");
    setNotice("");

    try {
      const result = await api<EvaluationResponse>("/evaluation/run", {
        method: "POST",
      });

      setEvaluation({
        available: true,
        status: result.status,
        message: result.message,
        report: result.report,
      });

      setNotice("Retrieval evaluation completed successfully.");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Evaluation failed. Check the backend terminal.",
      );
    } finally {
      setRunningEvaluation(false);
    }
  }

  async function refreshMonitor() {
    if (refreshingStatus) return;

    setRefreshingStatus(true);

    try {
      await Promise.all([
        refreshHealth(),
        refreshDocuments(),
        refreshEvaluation(),
      ]);
      setNotice("Dashboard status refresh completed.");
      setError("");
    } finally {
      setRefreshingStatus(false);
    }
  }

  function startNewQuestion() {
    setAnswer(null);
    setQuestion("");
    setError("");
    setNotice("");

    window.requestAnimationFrame(() => {
      questionInput.current?.focus();
    });
  }

  const filteredDocuments = documents.filter((document) =>
    displayName(document)
      .toLowerCase()
      .includes(documentSearch.trim().toLowerCase()),
  );

  const totalChunks = documents.reduce(
    (total, document) =>
      total +
      (typeof document.chunk_count === "number" ? document.chunk_count : 0),
    0,
  );

  const metrics = evaluation?.report?.metrics;
  const report = evaluation?.report;

  return (
    <div className="app-shell">
      <aside
        id="nexus-navigation"
        className={`sidebar ${mobileNavOpen ? "sidebar-open" : ""}`}
        aria-label="Workspace sidebar"
      >
        <div className="brand">
          <LogoMark size={43} />

          <div className="brand-copy">
            <div className="brand-name">
              NEXUS<span>.</span>
            </div>
            <div className="brand-caption">KNOWLEDGE INTELLIGENCE</div>
          </div>

          <button
            type="button"
            className="icon-button mobile-close"
            aria-label="Close navigation"
            title="Close navigation"
            onClick={() => setMobileNavOpen(false)}
          >
            <X size={18} />
          </button>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="nav-list" aria-label="Main navigation">
          {NAV_ITEMS.map(({ label, description, icon: Icon }) => (
            <button
              type="button"
              key={label}
              className={`nav-item ${section === label ? "active" : ""}`}
              onClick={() => navigate(label)}
              aria-current={section === label ? "page" : undefined}
            >
              <span className="nav-icon">
                <Icon size={18} strokeWidth={1.8} />
              </span>
              <span className="nav-copy">
                <strong>{label}</strong>
                <small>{description}</small>
              </span>
              {section === label && (
                <ChevronRight className="nav-chevron" size={16} />
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="local-card">
            <div className="local-card-icon">
              <ShieldCheck size={19} />
            </div>
            <div>
              <strong>Private by design</strong>
              <p>Your research stays in your local workspace.</p>
            </div>
          </div>

          <div className="sidebar-footer">
            <span className={`status-dot ${health ? "online" : ""}`} />
            <span>{health ? "API connected" : "API unavailable"}</span>
            <span className="version-tag">v0.1.0</span>
          </div>
        </div>
      </aside>

      {mobileNavOpen && (
        <button
          type="button"
          className="mobile-backdrop"
          aria-label="Close navigation overlay"
          onClick={() => setMobileNavOpen(false)}
        />
      )}

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="icon-button mobile-menu"
              aria-label="Open navigation"
              aria-controls="nexus-navigation"
              aria-expanded={mobileNavOpen}
              onClick={() => setMobileNavOpen(true)}
            >
              <Menu size={19} />
            </button>

            <div className="breadcrumbs">
              <span>Workspace</span>
              <ChevronRight size={14} />
              <strong>{section}</strong>
            </div>
          </div>

          <div className="topbar-right">
            <span className={`status-pill ${health ? "healthy" : "offline"}`}>
              <span className="status-dot" />
              {health ? "System operational" : "Waiting for API"}
            </span>
            <div className="avatar" aria-label="NEXUS workspace">
              NX
            </div>
          </div>
        </header>

        <div className="page-content">
          {error && (
            <div className="alert error-alert" role="alert">
              <CircleHelp size={17} />
              <span>{error}</span>
              <button
                type="button"
                aria-label="Dismiss error"
                onClick={() => setError("")}
              >
                <X size={16} />
              </button>
            </div>
          )}

          {notice && (
            <div className="alert success-alert" role="status">
              <CheckCircle2 size={17} />
              <span>{notice}</span>
              <button
                type="button"
                aria-label="Dismiss message"
                onClick={() => setNotice("")}
              >
                <X size={16} />
              </button>
            </div>
          )}

          {section === "Ask NEXUS" && (
            <>
              <section className="hero-heading">
                <div>
                  <div className="eyebrow">
                    <Sparkles size={14} />
                    YOUR RESEARCH WORKSPACE
                  </div>
                  <h1>Think with your knowledge.</h1>
                  <p>
                    Ask questions across your documents. Explore the evidence
                    behind every answer.
                  </p>
                </div>

                <div className="hero-status">
                  <span className="hero-status-icon">
                    <Zap size={17} />
                  </span>
                  <div>
                    <strong>Evidence-led research</strong>
                    <span>Retrieval augmented generation</span>
                  </div>
                </div>
              </section>

              <section className="query-panel">
                <div className="query-panel-heading">
                  <div className="panel-icon">
                    <MessageSquareText size={18} />
                  </div>
                  <div>
                    <strong>Start a research question</strong>
                    <span>Search the knowledge you have indexed.</span>
                  </div>
                  <span className="local-tag">
                    <span className="status-dot online" />
                    Local workspace
                  </span>
                </div>

                <form onSubmit={askQuestion}>
                  <label className="sr-only" htmlFor="research-question">
                    Research question
                  </label>
                  <textarea
                    ref={questionInput}
                    id="research-question"
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="Ask something about your papers, notes, or documents..."
                    rows={4}
                    maxLength={4000}
                  />

                  <div className="query-toolbar">
                    <label className="topk-control">
                      <span>Evidence passages</span>
                      <select
                        value={topK}
                        onChange={(event) =>
                          setTopK(Number(event.target.value))
                        }
                      >
                        {[3, 5, 8, 10, 15, 20].map((count) => (
                          <option key={count} value={count}>
                            {count}
                          </option>
                        ))}
                      </select>
                    </label>

                    <span className="character-count">
                      {question.length.toLocaleString()} / 4,000
                    </span>

                    <button
                      type="submit"
                      className="primary-button"
                      disabled={asking || !question.trim()}
                    >
                      {asking ? (
                        <LoaderCircle className="spin" size={17} />
                      ) : (
                        <Send size={16} />
                      )}
                      {asking ? "Researching" : "Ask NEXUS"}
                    </button>
                  </div>
                </form>
              </section>

              {!answer && !asking && (
                <section className="suggestion-section">
                  <div className="section-heading">
                    <div>
                      <h2>What would you like to discover?</h2>
                      <p>Choose a starting point or write your own question.</p>
                    </div>
                  </div>

                  <div className="suggestion-grid">
                    {SUGGESTIONS.map(
                      ({ title, description, question: suggestedQuestion, icon: Icon }) => (
                        <button
                          type="button"
                          className="suggestion-card"
                          key={title}
                          onClick={() => {
                            setQuestion(suggestedQuestion);
                            window.requestAnimationFrame(() => {
                              questionInput.current?.focus();
                            });
                          }}
                        >
                          <span className="suggestion-icon">
                            <Icon size={19} />
                          </span>
                          <span className="suggestion-copy">
                            <strong>{title}</strong>
                            <small>{description}</small>
                          </span>
                        </button>
                      ),
                    )}
                  </div>

                  <div className="research-note">
                    <ShieldCheck size={16} />
                    <span>
                      Answers are generated from retrieved context. Check the
                      source passages before relying on important conclusions.
                    </span>
                  </div>
                </section>
              )}

              {asking && (
                <div className="loading-state" role="status" aria-live="polite">
                  <div className="loading-orb">
                    <LoaderCircle className="spin" size={25} />
                  </div>
                  <strong>Exploring your knowledge base</strong>
                  <span>
                    Retrieving relevant passages and preparing your answer…
                  </span>
                </div>
              )}

              {answer && (
                <div className="results-layout">
                  <section className="answer-card">
                    <div className="result-heading">
                      <div className="panel-icon">
                        <Sparkles size={18} />
                      </div>
                      <div className="result-title">
                        <strong>NEXUS response</strong>
                        <span>Generated from retrieved context</span>
                      </div>
                      <span className="evidence-count">
                        <FileCheck2 size={14} />
                        {answer.source_count}{" "}
                        {answer.source_count === 1 ? "source" : "sources"}
                      </span>
                    </div>

                    <div className="question-echo">
                      <span>Your question</span>
                      <p>{answer.question}</p>
                    </div>

                    <div className="answer-text">{answer.answer}</div>

                    <div className="answer-disclaimer">
                      <ShieldCheck size={16} />
                      <span>
                        Verify important conclusions against the cited source
                        passages.
                      </span>
                    </div>
                  </section>

                  <section className="sources-section">
                    <div className="section-heading">
                      <div>
                        <h2>Retrieved evidence</h2>
                        <p>
                          Inspect the passages retrieved for this response.
                        </p>
                      </div>
                      <span className="count-badge">
                        {answer.sources.length}
                      </span>
                    </div>

                    {answer.sources.length === 0 ? (
                      <div className="inline-empty">
                        No source passages were returned for this query.
                      </div>
                    ) : (
                      answer.sources.map((source, index) => (
                        <article
                          className="source-card"
                          key={`${source.chunk_id || source.source}-${index}`}
                        >
                          <div className="source-topline">
                            <span className="citation-badge">
                              [{source.citation || `S${index + 1}`}]
                            </span>
                            <span className="source-name">
                              <FileText size={15} />
                              {displayName({ name: source.source })}
                            </span>
                            <span className="source-score">
                              {typeof source.similarity === "number"
                                ? `${(source.similarity * 100).toFixed(1)}% match`
                                : "Similarity unavailable"}
                            </span>
                          </div>

                          <p className="source-excerpt">
                            {source.text || "No passage text available."}
                          </p>

                          <div className="source-meta">
                            <span>
                              {source.page_number != null
                                ? `Page ${source.page_number}`
                                : "Page not specified"}
                            </span>
                            {source.chunk_id && <span>{source.chunk_id}</span>}
                          </div>
                        </article>
                      ))
                    )}
                  </section>

                  <button
                    type="button"
                    className="secondary-button new-question-button"
                    onClick={startNewQuestion}
                  >
                    <Plus size={16} />
                    Start a new question
                  </button>
                </div>
              )}
            </>
          )}

          {section === "Knowledge Base" && (
            <>
              <section className="page-heading">
                <div>
                  <div className="eyebrow">
                    <BookOpen size={14} />
                    DOCUMENT LIBRARY
                  </div>
                  <h1>Your knowledge, organized.</h1>
                  <p>
                    Manage the PDFs NEXUS uses to retrieve evidence and answer
                    questions.
                  </p>
                </div>

                <button
                  type="button"
                  className="primary-button"
                  onClick={() => fileInput.current?.click()}
                  disabled={uploading}
                >
                  {uploading ? (
                    <LoaderCircle className="spin" size={17} />
                  ) : (
                    <Plus size={17} />
                  )}
                  {uploading ? "Processing PDF" : "Add document"}
                </button>

                <input
                  ref={fileInput}
                  className="hidden-input"
                  type="file"
                  accept=".pdf,application/pdf"
                  onChange={uploadPdf}
                />
              </section>

              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-topline">
                    <span className="stat-icon">
                      <FolderOpen size={19} />
                    </span>
                    <span className="stat-trend">Document registry</span>
                  </div>
                  <span className="stat-label">Indexed documents</span>
                  <strong>{documents.length}</strong>
                  <span className="stat-foot">
                    Documents returned by the API
                  </span>
                </div>

                <div className="stat-card">
                  <div className="stat-topline">
                    <span className="stat-icon">
                      <Network size={19} />
                    </span>
                    <span className="stat-trend">Indexed content</span>
                  </div>
                  <span className="stat-label">Registered chunks</span>
                  <strong>{totalChunks.toLocaleString()}</strong>
                  <span className="stat-foot">
                    Sum of available document metadata
                  </span>
                </div>

                <div className="stat-card">
                  <div className="stat-topline">
                    <span className="stat-icon">
                      <Activity size={19} />
                    </span>
                    <span className={`stat-state ${health ? "is-online" : ""}`}>
                      <span className="status-dot" />
                      {health ? "Connected" : "Offline"}
                    </span>
                  </div>
                  <span className="stat-label">API status</span>
                  <strong className={health ? "green-text" : "red-text"}>
                    {health ? "Healthy" : "Unavailable"}
                  </strong>
                  <span className="stat-foot">Live health endpoint</span>
                </div>
              </div>

              <section className="library-panel">
                <div className="library-toolbar">
                  <div>
                    <h2>Indexed documents</h2>
                    <p>Search and manage your current document registry.</p>
                  </div>
                  <button
                    type="button"
                    className="icon-button"
                    title="Refresh documents"
                    aria-label="Refresh documents"
                    onClick={() => void refreshDocuments()}
                    disabled={loadingDocuments}
                  >
                    <RefreshCw
                      size={17}
                      className={loadingDocuments ? "spin" : ""}
                    />
                  </button>
                </div>

                <div className="library-search">
                  <Search size={17} />
                  <input
                    value={documentSearch}
                    onChange={(event) => setDocumentSearch(event.target.value)}
                    placeholder="Find a document by filename..."
                    aria-label="Search documents"
                  />
                  {documentSearch && (
                    <button
                      type="button"
                      aria-label="Clear document search"
                      title="Clear search"
                      onClick={() => setDocumentSearch("")}
                    >
                      <X size={15} />
                    </button>
                  )}
                </div>

                {loadingDocuments ? (
                  <div className="loading-state compact" role="status">
                    <LoaderCircle className="spin" size={22} />
                    <span>Loading your document library…</span>
                  </div>
                ) : filteredDocuments.length === 0 ? (
                  <div className="empty-state library-empty">
                    <div className="empty-icon">
                      {documents.length > 0 ? (
                        <Search size={23} />
                      ) : (
                        <Upload size={23} />
                      )}
                    </div>
                    <h3>
                      {documents.length > 0
                        ? "No matching documents"
                        : "Your library starts here."}
                    </h3>
                    <p>
                      {documents.length > 0
                        ? "Try another search term to find a registered document."
                        : "Upload a PDF to make its contents searchable in your research workspace."}
                    </p>
                    {documents.length === 0 && (
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => fileInput.current?.click()}
                        disabled={uploading}
                      >
                        <Plus size={16} />
                        Choose a PDF
                      </button>
                    )}
                  </div>
                ) : (
                  <div className="document-list">
                    {filteredDocuments.map((document, index) => {
                      const path = document.source_path;
                      const isDeleting =
                        path != null && deletingDocument === path;

                      return (
                        <article
                          className="document-row"
                          key={`${path || displayName(document)}-${index}`}
                        >
                          <div className="document-icon">
                            <FileText size={20} />
                          </div>

                          <div className="document-info">
                            <strong title={displayName(document)}>
                              {displayName(document)}
                            </strong>
                            <span>
                              <Check size={13} />
                              {String(
                                document.status || "Registered in index",
                              )}
                            </span>
                          </div>

                          <div className="document-details">
                            {typeof document.page_count === "number" && (
                              <span>
                                {document.page_count}{" "}
                                {document.page_count === 1 ? "page" : "pages"}
                              </span>
                            )}
                            {typeof document.chunk_count === "number" && (
                              <span>{document.chunk_count} chunks</span>
                            )}
                          </div>

                          <div className="document-actions">
                            {isDeleting ? (
                              <LoaderCircle
                                className="spin"
                                size={18}
                                aria-label="Deleting document"
                              />
                            ) : (
                              <CheckCircle2
                                className="document-check"
                                size={18}
                                aria-label="Indexed"
                              />
                            )}

                            <button
                              type="button"
                              className="icon-button danger-icon-button"
                              title={`Remove ${displayName(document)}`}
                              aria-label={`Remove ${displayName(document)}`}
                              disabled={!path || deletingDocument !== null}
                              onClick={() => void deleteDocument(document)}
                            >
                              <Trash2 size={16} />
                            </button>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                )}

                <div className="library-footer">
                  <span>
                    Showing {filteredDocuments.length} of {documents.length}{" "}
                    documents
                  </span>
                  <span>
                    <ShieldCheck size={14} />
                    Managed in your local workspace
                  </span>
                </div>
              </section>
            </>
          )}

          {section === "System Monitor" && (
            <>
              <section className="page-heading">
                <div>
                  <div className="eyebrow">
                    <Activity size={14} />
                    SYSTEM OBSERVABILITY
                  </div>
                  <h1>Know how NEXUS is performing.</h1>
                  <p>
                    Inspect API availability, indexed content and measured
                    retrieval quality.
                  </p>
                </div>

                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => void refreshMonitor()}
                  disabled={refreshingStatus}
                >
                  <RefreshCw
                    size={16}
                    className={refreshingStatus ? "spin" : ""}
                  />
                  {refreshingStatus ? "Refreshing…" : "Refresh status"}
                </button>
              </section>

              <div className="monitor-grid">
                <article className="monitor-card">
                  <div className="monitor-card-top">
                    <span className="monitor-icon">
                      <Activity size={20} />
                    </span>
                    <span
                      className={`status-pill ${health ? "healthy" : "offline"}`}
                    >
                      <span className="status-dot" />
                      {health ? "Connected" : "Offline"}
                    </span>
                  </div>
                  <span className="stat-label">API availability</span>
                  <strong className={health ? "green-text" : "red-text"}>
                    {health ? "Operational" : "Unavailable"}
                  </strong>
                  <p>{health?.service || "NEXUS API"}</p>
                  <div className="monitor-card-foot">
                    <Clock3 size={14} />
                    Health check every 15 seconds
                  </div>
                </article>

                <article className="monitor-card">
                  <div className="monitor-card-top">
                    <span className="monitor-icon">
                      <BookOpen size={20} />
                    </span>
                    <span className="monitor-neutral">Current registry</span>
                  </div>
                  <span className="stat-label">Indexed documents</span>
                  <strong>{documents.length}</strong>
                  <p>
                    {totalChunks.toLocaleString()} registered chunks, based on
                    available document metadata.
                  </p>
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => navigate("Knowledge Base")}
                  >
                    View document library
                    <ChevronRight size={14} />
                  </button>
                </article>

                <article className="monitor-card">
                  <div className="monitor-card-top">
                    <span className="monitor-icon">
                      <Gauge size={20} />
                    </span>
                    <span className="monitor-neutral">Measured results</span>
                  </div>
                  <span className="stat-label">Latest Recall@5</span>
                  <strong>
                    {metrics
                      ? `${(metrics.recall_at_5 * 100).toFixed(1)}%`
                      : "—"}
                  </strong>
                  <p>
                    {metrics
                      ? `${metrics.question_count} evaluation questions`
                      : "Run retrieval evaluation to measure quality."}
                  </p>
                  <span className="monitor-card-foot">
                    {metrics
                      ? "From the latest evaluation report"
                      : "No metric available"}
                  </span>
                </article>
              </div>

              <section className="evaluation-panel">
                <div className="evaluation-heading">
                  <div>
                    <div className="eyebrow">
                      <Sparkles size={14} />
                      RETRIEVAL QUALITY
                    </div>
                    <h2>Evaluation results</h2>
                    <p>
                      Measure whether relevant passages appear near the top of
                      retrieval results.
                    </p>
                  </div>

                  <button
                    type="button"
                    className="primary-button"
                    onClick={() => void runEvaluation()}
                    disabled={runningEvaluation}
                  >
                    {runningEvaluation ? (
                      <LoaderCircle size={16} className="spin" />
                    ) : (
                      <RefreshCw size={16} />
                    )}
                    {runningEvaluation ? "Evaluating…" : "Run evaluation"}
                  </button>
                </div>

                {loadingEvaluation ? (
                  <div className="evaluation-message" role="status">
                    <LoaderCircle className="spin" size={18} />
                    Loading the latest evaluation report…
                  </div>
                ) : !evaluation?.available || !report || !metrics ? (
                  <div className="evaluation-empty">
                    <div className="empty-icon">
                      <Gauge size={22} />
                    </div>
                    <strong>No evaluation report available</strong>
                    <p>
                      {evaluation?.message ||
                        "Run retrieval evaluation to generate metrics from the configured dataset."}
                    </p>
                    <p>
                      Metrics appear here only when the backend returns a
                      valid report.
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="evaluation-meta">
                      <span className="report-name">
                        <FileCheck2 size={15} />
                        {report.dataset_name}
                      </span>
                      <span>Version {report.dataset_version || "—"}</span>
                      <span>{report.indexed_chunk_count} indexed chunks</span>
                      <span>Top K: {report.top_k}</span>
                      <span>Updated {formatDate(report.evaluated_at_utc)}</span>
                    </div>

                    <div className="evaluation-metrics">
                      {[
                        {
                          label: "Recall@1",
                          value: metrics.recall_at_1,
                          note: "Relevant in top 1",
                        },
                        {
                          label: "Recall@3",
                          value: metrics.recall_at_3,
                          note: "Relevant in top 3",
                        },
                        {
                          label: "Recall@5",
                          value: metrics.recall_at_5,
                          note: "Relevant in top 5",
                        },
                        {
                          label: "MRR",
                          value: metrics.mrr,
                          note: "Ranking quality",
                        },
                      ].map((metric) => (
                        <div className="evaluation-metric" key={metric.label}>
                          <span>{metric.label}</span>
                          <strong>
                            {(metric.value * 100).toFixed(1)}%
                          </strong>
                          <div
                            className="metric-track"
                            role="progressbar"
                            aria-label={metric.label}
                            aria-valuemin={0}
                            aria-valuemax={100}
                            aria-valuenow={Math.round(
                              Math.max(0, Math.min(1, metric.value)) * 100,
                            )}
                          >
                            <span
                              style={{
                                width: `${Math.max(
                                  0,
                                  Math.min(100, metric.value * 100),
                                )}%`,
                              }}
                            />
                          </div>
                          <small>{metric.note}</small>
                        </div>
                      ))}
                    </div>

                    <div className="evaluation-summary">
                      <span>
                        <CheckCircle2 size={15} />
                        {metrics.question_count} questions evaluated
                      </span>
                      <span>
                        {metrics.questions_with_missing_labels} questions with
                        missing labels
                      </span>
                      <span>
                        {metrics.questions_with_no_relevant_result} questions
                        with no relevant result
                      </span>
                    </div>

                    <section className="evaluation-questions">
                      <button
                        type="button"
                        className="evaluation-table-heading"
                        onClick={() =>
                          setEvaluationQuestionsOpen((open) => !open)
                        }
                        aria-expanded={evaluationQuestionsOpen}
                        aria-controls="evaluation-question-list"
                      >
                        <span>
                          <h3>Question-level results</h3>
                          <span className="evaluation-table-description">
                            Individual retrieval outcomes from the report.
                          </span>
                        </span>
                        <ChevronDown
                          size={17}
                          className={
                            evaluationQuestionsOpen ? "chevron-open" : ""
                          }
                        />
                      </button>

                      {evaluationQuestionsOpen && (
                        <div id="evaluation-question-list">
                          {(report.questions || []).length === 0 ? (
                            <div className="inline-empty">
                              This report does not contain question-level
                              results.
                            </div>
                          ) : (
                            (report.questions || []).map((item) => (
                              <article
                                className="evaluation-question"
                                key={item.id}
                              >
                                <div className="evaluation-question-main">
                                  <strong>{item.question}</strong>
                                  <span>{item.id}</span>
                                </div>

                                <div className="evaluation-question-result">
                                  <span
                                    className={`evaluation-result-pill ${
                                      item.first_relevant_rank == null
                                        ? "evaluation-failed"
                                        : "evaluation-passed"
                                    }`}
                                  >
                                    {item.first_relevant_rank == null
                                      ? "Not found"
                                      : `First relevant: #${item.first_relevant_rank}`}
                                  </span>
                                  <small>
                                    Recall@5:{" "}
                                    {(item.recall_at_5 * 100).toFixed(0)}%
                                  </small>
                                </div>
                              </article>
                            ))
                          )}
                        </div>
                      )}
                    </section>
                  </>
                )}
              </section>

              <section className="info-panel">
                <div className="info-panel-icon">
                  <ShieldCheck size={19} />
                </div>
                <div>
                  <h2>Monitoring scope</h2>
                  <p>
                    API availability is checked live. Retrieval metrics come
                    from the latest evaluation report. Model readiness and
                    query latency are not presented as live measurements
                    because those metrics are not currently supplied by the
                    backend.
                  </p>
                </div>
              </section>
            </>
          )}
        </div>

        <footer className="page-footer">
          <div className="footer-brand">
            <LogoMark size={24} />
            <span>
              <strong>NEXUS</strong>
              <span className="footer-separator">/</span>
              Research &amp; Knowledge Intelligence
            </span>
          </div>
          <span className="footer-connection">
            <span className={`status-dot ${health ? "online" : ""}`} />
            {health ? "Connected to local API" : "API connection unavailable"}
          </span>
        </footer>
      </main>
    </div>
  );
}

export default App;
