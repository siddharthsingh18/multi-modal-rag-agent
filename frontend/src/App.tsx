import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowUp,
  BookOpen,
  Check,
  ChevronRight,
  FileText,
  KeyRound,
  LoaderCircle,
  Paperclip,
  Plus,
  Search,
  Settings2,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { getHealth, ingestFile, ingestText, queryRag, type QueryResult, type Source } from "./api";

type Message = { role: "user" | "assistant"; content: string; sources?: Source[]; score?: number };

const starterText = `Python is a high-level programming language known for its simple syntax and readability. It is commonly used for web development, data science, machine learning, automation, scientific computing, and scripting.\n\nFastAPI is a modern Python framework for building web APIs. It supports asynchronous programming, automatic API documentation, request validation, and high performance.\n\nRetrieval-Augmented Generation, or RAG, combines document search with large language models. A RAG system first retrieves relevant document sections and then uses those sections to generate a more accurate answer.`;

function App() {
  const [apiKey, setApiKey] = useState(import.meta.env.VITE_API_KEY || "development");
  const [text, setText] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [notice, setNotice] = useState("");
  const [health, setHealth] = useState<"checking" | "healthy" | "degraded" | "offline">("checking");
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    getHealth().then((result) => setHealth(result.status === "healthy" ? "healthy" : "degraded")).catch(() => setHealth("offline"));
  }, []);

  const hasKnowledge = messages.length > 0 || Boolean(notice);
  const latestSources = useMemo(() => sources.slice(0, 4), [sources]);

  async function handleIngest() {
    if (!text.trim()) return;
    setIsIngesting(true);
    setNotice("");
    try {
      const result = await ingestText(text, apiKey);
      setNotice(`Knowledge added · ${result.num_chunks} chunk${result.num_chunks === 1 ? "" : "s"} indexed`);
      setText("");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not index this text");
    } finally {
      setIsIngesting(false);
    }
  }

  async function handleFile(file?: File) {
    if (!file) return;
    setIsIngesting(true);
    setNotice("");
    try {
      const result = await ingestFile(file, apiKey);
      setNotice(`${file.name} indexed · ${result.num_chunks} chunk${result.num_chunks === 1 ? "" : "s"}`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not upload this file");
    } finally {
      setIsIngesting(false);
    }
  }

  async function handleAsk(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim() || isQuerying) return;
    const currentQuestion = question.trim();
    setQuestion("");
    setMessages((current) => [...current, { role: "user", content: currentQuestion }]);
    setIsQuerying(true);
    try {
      const result: QueryResult = await queryRag(currentQuestion, apiKey);
      setSources(result.documents);
      setMessages((current) => [...current, { role: "assistant", content: result.answer, sources: result.documents, score: result.reflection?.score }]);
    } catch (error) {
      setMessages((current) => [...current, { role: "assistant", content: error instanceof Error ? error.message : "The query could not be completed." }]);
    } finally {
      setIsQuerying(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark"><Sparkles size={16} /></span><span>ATLAS<span className="brand-muted"> / RAG</span></span></div>
        <div className="sidebar-section-label">Workspace</div>
        <button className="nav-item active"><Search size={17} /> Ask your knowledge base</button>
        <button className="nav-item"><BookOpen size={17} /> Library <span className="nav-count">01</span></button>
        <div className="sidebar-spacer" />
        <div className="sidebar-card">
          <div className="eyebrow">Current collection</div>
          <strong>Workspace memory</strong>
          <div className="collection-progress"><span /></div>
          <small>{hasKnowledge ? "1 source indexed" : "Waiting for your first source"}</small>
        </div>
        <button className="nav-item" onClick={() => setShowSettings(true)}><Settings2 size={17} /> Settings</button>
        <div className="profile"><div className="avatar">YK</div><div><strong>Your workspace</strong><small>Local environment</small></div><ChevronRight size={15} /></div>
      </aside>

      <main className="main-content">
        <header className="topbar"><div><div className="eyebrow">Research workspace / 01</div><h1>Ask your knowledge base</h1></div><div className="topbar-actions"><div className={`health-pill ${health}`}><span className="health-dot" /> {health === "checking" ? "Checking systems" : health === "offline" ? "API offline" : `Systems ${health}`}</div><button className="icon-button" onClick={() => setShowSettings(true)} aria-label="Open settings"><Settings2 size={18} /></button></div></header>
        <section className="workspace-grid">
          <div className="conversation-panel">
            <div className="context-strip"><div className="context-icon"><Activity size={17} /></div><div><strong>Grounded mode</strong><span>Answers are generated from your indexed sources.</span></div><span className="live-tag">LIVE</span></div>
            <div className="chat-area">
              {messages.length === 0 ? <div className="empty-state"><div className="orbital-mark"><Sparkles size={28} /></div><p className="kicker">Your research desk</p><h2>Make your sources<br /><em>useful.</em></h2><p className="empty-copy">Add a document or paste a note on the left, then ask a question here. Atlas will retrieve the most relevant passages and show its work.</p><div className="suggestion-row"><button onClick={() => setQuestion("What are the key ideas in these sources?")}>Summarize my sources <ArrowUp size={13} /></button><button onClick={() => setQuestion("How does RAG work?")}>Explain the main concept <ArrowUp size={13} /></button></div></div> : <div className="messages">{messages.map((message, index) => <div className={`message ${message.role}`} key={`${message.role}-${index}`}><div className="message-label">{message.role === "user" ? "You" : "Atlas"}</div><div className="message-content">{message.content}</div>{message.score && <div className="confidence"><Check size={13} /> Reflection score {message.score}/10</div>}</div>)}{isQuerying && <div className="message assistant"><div className="message-label">Atlas</div><div className="thinking"><LoaderCircle size={16} className="spin" /> Searching your sources</div></div>}</div>}
            </div>
            <form className="ask-form" onSubmit={handleAsk}><textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about your sources..." rows={2} /><div className="ask-form-footer"><span><Paperclip size={15} /> Grounded answer</span><button className="send-button" type="submit" disabled={!question.trim() || isQuerying}><ArrowUp size={18} /></button></div></form>
          </div>
          <aside className="source-panel"><div className="panel-heading"><div><div className="eyebrow">Knowledge base</div><h3>Sources</h3></div><button className="plus-button" onClick={() => setText(starterText)} aria-label="Add sample text"><Plus size={17} /></button></div><div className="source-list">{latestSources.length ? latestSources.map((source, index) => <article className="source-card" key={source.id}><div className="source-card-top"><span className="source-number">0{index + 1}</span><span className="score">{Math.round(source.score * 100)}% match</span></div><p>{source.content}</p><small>Retrieved chunk · {source.id.slice(0, 8)}</small></article>) : <div className="no-sources"><FileText size={20} /><strong>No sources yet</strong><span>Indexed passages will appear here after your first question.</span></div>}</div><div className="source-footer"><span>{latestSources.length} retrieved passages</span><span className="footer-rule" /></div></aside>
        </section>
      </main>

      <aside className="intake-drawer"><div className="drawer-header"><div><div className="eyebrow">Add context</div><h2>Source intake</h2></div><button className="icon-button" onClick={() => setShowSettings(true)} aria-label="Open API settings"><KeyRound size={17} /></button></div><label className="dropzone"><input type="file" accept=".pdf,.txt,.md,.html" onChange={(event) => handleFile(event.target.files?.[0])} /><Upload size={20} /><strong>Drop a document here</strong><span>PDF, TXT, MD, HTML · max 10 MB</span></label><div className="or-divider"><span>or paste text</span></div><textarea className="intake-textarea" value={text} onChange={(event) => setText(event.target.value)} placeholder="Paste an excerpt, meeting note, or research brief..." /><button className="index-button" onClick={handleIngest} disabled={!text.trim() || isIngesting}>{isIngesting ? <LoaderCircle size={16} className="spin" /> : <Plus size={16} />} {isIngesting ? "Indexing source" : "Index source"}</button>{notice && <div className={`notice ${notice.includes("Could") || notice.includes("failed") ? "error" : "success"}`}><Check size={15} /> {notice}</div>}<div className="drawer-note"><Sparkles size={15} /><span>Atlas keeps retrieval transparent. Every answer includes the passages that informed it.</span></div></aside>

      {showSettings && <div className="modal-backdrop" onClick={() => setShowSettings(false)}><div className="settings-modal" onClick={(event) => event.stopPropagation()}><div className="modal-title"><div><div className="eyebrow">Workspace settings</div><h2>Connection</h2></div><button className="icon-button" onClick={() => setShowSettings(false)} aria-label="Close settings"><X size={18} /></button></div><label className="field-label">API key<input type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="X-API-Key value" /></label><p className="modal-help">Stored only in this browser session. The backend uses this key to protect query and ingestion routes.</p><button className="index-button" onClick={() => setShowSettings(false)}>Save connection</button></div></div>}
    </div>
  );
}

export default App;
