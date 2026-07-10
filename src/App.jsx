import { useState, useEffect, useCallback, useRef } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine, ComposedChart } from "recharts";

/* ═══ UTILS ═══ */
function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function generateWeeklyData(weeks, seed, shocks) {
  const rng = mulberry32(seed); const data = []; let idx = 100; const s = new Date(2020, 0, 6);
  for (let w = 0; w < weeks; w++) {
    idx += (100 - idx) * 0.02 + (rng() - 0.5) * 1.8 + (shocks.find(x => x.w === w)?.mag || 0);
    idx = Math.max(85, Math.min(115, idx));
    const d = new Date(s); d.setDate(d.getDate() + w * 7);
    data.push({ date: d.toISOString().slice(0, 10), index: Math.round(idx * 100) / 100 });
  }
  for (let i = 0; i < data.length; i++) {
    const w = data.slice(Math.max(0, i - 29), i + 1);
    data[i].ma30 = Math.round(w.reduce((s, d) => s + d.index, 0) / w.length * 100) / 100;
    data[i].devFromMA = Math.round((data[i].index / data[i].ma30 - 1) * 1e4) / 100;
    data[i].weeklyChg = i > 0 ? Math.round((data[i].index / data[i - 1].index - 1) * 1e4) / 100 : 0;
    data[i].monthlyChg = i >= 4 ? Math.round((data[i].index / data[i - 4].index - 1) * 1e4) / 100 : 0;
    if (i >= 20) { const r = []; for (let j = i - 19; j <= i; j++) r.push(data[j].weeklyChg || 0); const m = r.reduce((a, b) => a + b, 0) / r.length; data[i].vol20w = Math.round(Math.sqrt(r.reduce((a, b) => a + (b - m) ** 2, 0) / r.length) * 100) / 100; }
  }
  return data;
}

/* ── Synthetic fallback data (used when dashboard_data.json not available) ── */
const W = 278;
const SYNTH = {
  AAPL: { name: "Apple", data: generateWeeklyData(W, 101, [{ w: 10, mag: -6 }, { w: 142, mag: 4 }, { w: 170, mag: 3 }, { w: 230, mag: 3 }, { w: 260, mag: -3 }]) },
  GOOGL: { name: "Alphabet", data: generateWeeklyData(W, 202, [{ w: 10, mag: -5 }, { w: 28, mag: -3 }, { w: 156, mag: -5 }, { w: 158, mag: -6 }, { w: 172, mag: 3 }, { w: 240, mag: -4 }]) },
  META: { name: "Meta Platforms", data: generateWeeklyData(W, 303, [{ w: 10, mag: -5 }, { w: 92, mag: -7 }, { w: 96, mag: -3 }, { w: 120, mag: -5 }, { w: 148, mag: -6 }, { w: 200, mag: 5 }, { w: 250, mag: -2 }]) },
  MSFT: { name: "Microsoft", data: generateWeeklyData(W, 404, [{ w: 10, mag: -4 }, { w: 108, mag: 3 }, { w: 150, mag: 5 }, { w: 156, mag: -3 }, { w: 210, mag: 4 }]) },
  AMZN: { name: "Amazon", data: generateWeeklyData(W, 505, [{ w: 10, mag: -4 }, { w: 56, mag: -2 }, { w: 148, mag: -3 }, { w: 265, mag: -2 }]) },
};
const SYNTH_SPILL = [{ t: "META", v: 6.9 }, { t: "AAPL", v: 3.7 }, { t: "AMZN", v: -1.7 }, { t: "MSFT", v: -2.0 }, { t: "GOOGL", v: -6.9 }];

/* ── Convert pipeline JSON to frontend format ── */
function convertPipelineData(json) {
  const pre = {};
  for (const [ticker, info] of Object.entries(json.tickers || {})) {
    pre[ticker] = {
      name: info.name,
      data: (info.history || []).map(h => ({
        date: h.date,
        index: h.index,
        ma30: h.ma30,
        devFromMA: h.dev_from_ma,
        weeklyChg: h.weekly_chg,
        monthlyChg: 0, // computed below
        vol20w: h.vol_20w,
      })),
    };
    // Compute monthlyChg
    const d = pre[ticker].data;
    for (let i = 4; i < d.length; i++) {
      d[i].monthlyChg = d[i - 4].index ? Math.round((d[i].index / d[i - 4].index - 1) * 1e4) / 100 : 0;
    }
  }

  const spillNet = json.spillover?.net || {};
  const spill = Object.entries(spillNet)
    .map(([t, v]) => ({ t, v }))
    .sort((a, b) => b.v - a.v);

  return { pre, spill, tsi: json.spillover?.total_spillover_index || 0, profiles: json.profiles || {} };
}
const TCOL = { AAPL: "#8b8b8b", GOOGL: "#60a5fa", META: "#f87171", MSFT: "#34d399", AMZN: "#fbbf24" };
const PROFILES_STATIC = {
  AAPL: { domain: "apple.com", name: "Apple Inc.", marketCap: "$3.5T", summary: "Consumer electronics and software giant known for iPhone, Mac, and its premium ecosystem.", industry: "Consumer Electronics", keywords: ["hardware", "iOS", "services", "privacy", "luxury brand"] },
  GOOGL: { domain: "google.com", name: "Alphabet Inc.", marketCap: "$2.2T", summary: "Dominant search and digital advertising company, increasingly pivoting to AI and cloud.", industry: "Digital Advertising / AI", keywords: ["search", "cloud", "Android", "AI/ML", "antitrust"] },
  META: { domain: "meta.com", name: "Meta Platforms Inc.", marketCap: "$1.6T", summary: "Social media conglomerate operating Facebook, Instagram, WhatsApp, investing heavily in the metaverse.", industry: "Social Media", keywords: ["social platforms", "advertising", "metaverse", "VR/AR", "content moderation"] },
  MSFT: { domain: "microsoft.com", name: "Microsoft Corp.", marketCap: "$3.2T", summary: "Enterprise software and cloud leader, strategic partner to OpenAI and the largest software company by revenue.", industry: "Enterprise Software / Cloud", keywords: ["Azure", "Office 365", "OpenAI", "gaming", "enterprise"] },
  AMZN: { domain: "amazon.com", name: "Amazon.com Inc.", marketCap: "$2.1T", summary: "E-commerce and cloud infrastructure leader with growing logistics, streaming, and AI ambitions.", industry: "E-Commerce / Cloud", keywords: ["AWS", "e-commerce", "logistics", "Prime", "Alexa"] },
};
function LogoImg({ domain, name, size }) {
  const sz = size || 44;
  const [failed, setFailed] = useState(false);
  if (!domain || failed) return (<div style={{ width: sz, height: sz, borderRadius: 10, background: "rgba(96,165,250,0.12)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: sz * 0.45, fontWeight: 700, color: C.b, flexShrink: 0 }}>{(name || "?")[0].toUpperCase()}</div>);
  return (<img src={"https://logo.clearbit.com/" + domain + "?size=" + sz * 2} alt={name} style={{ width: sz, height: sz, borderRadius: 10, objectFit: "contain", background: "#fff", flexShrink: 0 }} onError={() => setFailed(true)} />);
}
const SPILL = [{ t: "META", v: 6.9 }, { t: "AAPL", v: 3.7 }, { t: "AMZN", v: -1.7 }, { t: "MSFT", v: -2.0 }, { t: "GOOGL", v: -6.9 }];
const HL = { "short-term": 7, "medium-term": 28, "long-term": 84 };
function decay(peak, dur, days) { return Math.round(peak * Math.exp(-days * 0.693 / (HL[dur] || 28)) * 100) / 100; }
const C = { bg: "#0b0b0d", card: "rgba(255,255,255,0.02)", border: "rgba(255,255,255,0.06)", bh: "rgba(16,185,129,0.25)", t1: "#f0efe8", t2: "#9ca3af", t3: "#4b5563", g: "#10b981", r: "#ef4444", a: "#f59e0b", p: "#a78bfa", b: "#60a5fa" };
function cc(v) { return v > 0.05 ? C.g : v < -0.05 ? C.r : C.t2; }
function fc(v) { return (v >= 0 ? "+" : "") + v.toFixed(2) + "%"; }
function fi(v) { return v.toFixed(2); }

async function askAI(msgs, search) {
  const r = await fetch("/api/anthropic", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: msgs, search: !!search }),
  });
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json())?.error || ""; } catch (e) {}
    throw new Error(detail || ("AI request failed (" + r.status + ")"));
  }
  return r.json();
}
function getText(d) { return (d?.content || []).filter(b => b.type === "text").map(b => b.text).join("\n"); }
function parseJSON(t) { try { const m = t.match(/[\[{][\s\S]*[\]}]/); return m ? JSON.parse(m[0]) : null; } catch(e) { return null; } }
function buildProjection(items) {
  const active = items.filter(n => n.impact?.peak_impact); if (!active.length) return [];
  return Array.from({ length: 85 }, (_, day) => {
    let pos = 0, neg = 0;
    active.forEach(n => { const r = decay(n.impact.peak_impact, n.impact.duration, (n.impact.days_ago || 0) + day); r > 0 ? pos += r : neg += r; });
    return { day, net: Math.round((pos + neg) * 100) / 100, positive: Math.round(pos * 100) / 100, negative: Math.round(neg * 100) / 100 };
  });
}

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (<div style={{ background: "#1a1a1e", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "8px 12px", fontSize: 11, fontFamily: "inherit" }}>
    <div style={{ color: C.t3, marginBottom: 4 }}>{label}</div>
    {payload.map((p, i) => (<div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
      <div style={{ width: 8, height: 2, background: p.stroke || p.color, borderRadius: 1 }} />
      <span style={{ color: p.stroke || p.color, fontWeight: 600 }}>{p.name}</span>
      <span style={{ color: C.t1, marginLeft: "auto", paddingLeft: 12 }}>{fi(p.value)}</span>
    </div>))}
  </div>);
}

/* ═══ MAIN ═══ */
export default function App() {
  /* ── Data source: try pipeline JSON, fallback to synthetic ── */
  const [PRE, setPRE] = useState(SYNTH);
  const [SPILL, setSPILL] = useState(SYNTH_SPILL);
  const [PROFILES, setPROFILES] = useState(PROFILES_STATIC);
  const [dataSource, setDataSource] = useState("loading");
  const [spillTSI, setSpillTSI] = useState(5.24);

  useEffect(() => {
    fetch("/dashboard_data.json")
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(json => {
        const { pre, spill, tsi, profiles } = convertPipelineData(json);
        if (Object.keys(pre).length > 0) {
          setPRE(pre);
          if (spill.length > 0) setSPILL(spill);
          if (tsi) setSpillTSI(tsi);
          if (Object.keys(profiles).length > 0) {
            const merged = { ...PROFILES_STATIC };
            for (const [k, v] of Object.entries(profiles)) {
              if (merged[k]) Object.assign(merged[k], v);
              else merged[k] = v;
            }
            setPROFILES(merged);
          }
          setDataSource("live");
        } else {
          setDataSource("demo");
        }
      })
      .catch(() => setDataSource("demo"));
  }, []);

  const [portfolio, setPortfolio] = useState(() => Object.keys(SYNTH).map(t => ({ ticker: t, name: SYNTH[t].name, custom: false })));
  /* Sync portfolio when data source changes */
  useEffect(() => {
    if (dataSource !== "loading") {
      setPortfolio(Object.keys(PRE).map(t => ({ ticker: t, name: PRE[t].name, custom: false })));
    }
  }, [dataSource, PRE]);

  const [editMode, setEditMode] = useState(false);
  const [sel, setSel] = useState("META");
  const [isCustom, setIsCustom] = useState(false);
  const [customName, setCustomName] = useState("");
  const [news, setNews] = useState([]);
  const [newsLoading, setNewsLoading] = useState(false);
  const [insights, setInsights] = useState(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [tab, setTab] = useState("overview");
  const [toast, setToast] = useState(null);
  const nRef = useRef({});
  const iRef = useRef({});
  const pRef = useRef({});
  const [profile, setProfile] = useState(null);

  const company = isCustom ? customName : (PRE[sel]?.name || sel);
  const hist = !isCustom ? PRE[sel] : null;
  const latest = hist?.data[hist.data.length - 1];
  const curProfile = !isCustom ? PROFILES[sel] : profile;
  const analyzed = news.filter(n => n.impact);
  const allDone = news.length > 0 && news.every(n => !n.loading) && analyzed.length > 0;
  const proj = allDone ? buildProjection(news) : [];
  const totalRes = analyzed.reduce((s, n) => s + (n.impact?.residual || 0), 0);

  const flash = (msg) => { setToast(msg); setTimeout(() => setToast(null), 2000); };
  const addPort = (t, n) => { if (portfolio.find(p => p.ticker === t)) { flash(n + " already added"); return; } setPortfolio(p => [...p, { ticker: t, name: n, custom: true }]); flash("+ " + n); };
  const rmPort = (t) => setPortfolio(p => p.filter(x => x.ticker !== t));

  const pick = (t, n) => { setIsCustom(!PRE[t]); setSel(t); if (!PRE[t]) setCustomName(n || t); nRef.current = {}; iRef.current = {}; pRef.current = {}; setNews([]); setInsights(null); setProfile(null); };
  const goCustom = (e) => { e.preventDefault(); if (!customName.trim()) return; setIsCustom(true); nRef.current = {}; iRef.current = {}; pRef.current = {}; setNews([]); setInsights(null); setProfile(null); setTab("company"); };

  /* Fetch profile for custom companies */
  useEffect(() => {
    if (!isCustom || !company || tab !== "company") return;
    const k = company.toLowerCase();
    if (pRef.current[k]) return;
    pRef.current[k] = true;
    askAI([{ role: "user", content: "Look up \"" + company + "\" company. Return ONLY JSON (no markdown): {\"name\":\"full legal name\",\"domain\":\"company main website domain like apple.com\",\"marketCap\":\"estimated market cap like $1.2T or $45B or Private\",\"summary\":\"1 sentence company description\",\"industry\":\"industry sector\",\"keywords\":[\"5 reputation-relevant keywords\"]}" }], true)
    .then(r => { const p = parseJSON(getText(r)); if (p) setProfile(p); })
    .catch(() => {});
  }, [isCustom, company, tab]);

  /* Fetch news + analyze + then auto-trigger insights */
  const fetchNews = useCallback(async (name) => {
    if (!name || nRef.current[name.toLowerCase()]) return;
    nRef.current[name.toLowerCase()] = true;
    setNewsLoading(true); setNews([]); setInsights(null); setInsightsLoading(false);
    try {
      const r = await askAI([{ role: "user", content: "Search for the 5 most recent news articles about \"" + name + "\" that could affect its corporate reputation. Return ONLY a JSON array sorted newest first (no markdown, no backticks): [{\"headline\":string,\"source\":string,\"date\":string,\"summary\":string,\"days_ago\":number}]" }], true);
      let arts = parseJSON(getText(r)) || [];
      arts.sort((a, b) => (a.days_ago || 0) - (b.days_ago || 0));
      if (!arts.length) { setNews([{ headline: "No recent news found", summary: "Try another company." }]); setNewsLoading(false); return; }
      setNews(arts.map(a => ({ ...a, impact: null, loading: true }))); setNewsLoading(false);

      const results = [];
      for (let idx = 0; idx < arts.length; idx++) {
        try {
          const ir = await askAI([{ role: "user", content: "Reputation analyst. News about " + name + ": \"" + arts[idx].headline + "\". " + arts[idx].summary + "\nSCALE: +/-5=existential, +/-3=major, +/-1.5=notable, +/-0.5=minor. Most is +/-0.5 to 1.5.\nReturn ONLY JSON: {\"peak_impact\":number,\"duration\":\"short-term\"|\"medium-term\"|\"long-term\",\"category\":\"Crisis\"|\"Negative\"|\"Neutral\"|\"Positive\"|\"Boost\",\"reasoning\":\"1 sentence\",\"stakeholders\":[\"group\"]}" }], false);
          let imp = parseJSON(getText(ir));
          if (imp) { const da = arts[idx].days_ago || 3; imp.days_ago = da; imp.residual = decay(imp.peak_impact, imp.duration, da); imp.halflife_days = HL[imp.duration] || 28; imp.decay_pct = imp.peak_impact ? Math.round((1 - Math.abs(imp.residual / imp.peak_impact)) * 100) : 0; }
          results.push({ ...arts[idx], impact: imp });
          setNews(prev => prev.map((x, j) => j === idx ? { ...x, impact: imp, loading: false } : x));
        } catch(e) {
          results.push({ ...arts[idx], impact: null });
          setNews(prev => prev.map((x, j) => j === idx ? { ...x, loading: false } : x));
        }
      }

      /* ── Now trigger insights directly ── */
      const analyzed = results.filter(n => n.impact);
      if (analyzed.length === 0) return;
      setInsightsLoading(true);
      const ctx = analyzed.map(n =>
        "- \"" + n.headline + "\" ("+n.impact.days_ago+"d ago) Peak:" + n.impact.peak_impact + " Current:" + n.impact.residual + " " + n.impact.duration + " " + n.impact.category
      ).join("\n");
      const tr = analyzed.reduce((s, n) => s + (n.impact?.residual || 0), 0);
      try {
        const insRes = await askAI([{ role: "user", content: "You are a senior reputation strategist at a global PR firm. Based on RECENT events for " + name + ", provide a strategic assessment.\n\nRECENT EVENTS:\n" + ctx + "\nNet residual impact: " + tr.toFixed(2) + "\n\nReturn ONLY JSON:\n{\"swot\":{\"strengths\":[\"str\",\"str\"],\"weaknesses\":[\"str\",\"str\"],\"opportunities\":[\"str\",\"str\"],\"threats\":[\"str\",\"str\"]},\"trajectory\":\"improving\"|\"stable\"|\"deteriorating\"|\"volatile\",\"trajectory_summary\":\"2 sentences on where reputation is heading\",\"recovery_plan\":{\"short_term\":{\"timeframe\":\"0-4 weeks\",\"key_events\":\"what recent events drive this\",\"actions\":[\"specific action 1\",\"specific action 2\"],\"target_audience\":\"who to focus on\",\"channels\":[\"channel\"],\"messaging\":\"1-2 sentence messaging direction\"},\"medium_term\":{\"timeframe\":\"1-3 months\",\"key_events\":\"str\",\"actions\":[\"str\"],\"target_audience\":\"str\",\"channels\":[\"str\"],\"messaging\":\"str\"},\"long_term\":{\"timeframe\":\"3-12 months\",\"key_events\":\"str\",\"actions\":[\"str\"],\"target_audience\":\"str\",\"channels\":[\"str\"],\"messaging\":\"str\"}},\"reputation_moat\":\"1 sentence on strongest reputation asset to lean into\"}" }], false);
        const parsed = parseJSON(getText(insRes));
        if (parsed) setInsights(parsed);
      } catch(e) {}
      setInsightsLoading(false);

    } catch(e) { setNewsLoading(false); }
  }, []);

  useEffect(() => { if (tab === "company" || tab === "news") fetchNews(company); }, [tab, company, fetchNews]);

  const impColor = (v) => v > 0.3 ? C.g : v < -0.3 ? C.r : C.t2;
  const sevColor = (s) => s === "high" ? C.r : s === "medium" ? C.a : C.t2;
  const sevBg = (s) => s === "high" ? "rgba(239,68,68,0.12)" : s === "medium" ? "rgba(245,158,11,0.12)" : "rgba(107,114,128,0.08)";
  const catColor = (c) => c === "Crisis" ? C.r : c === "Negative" ? C.a : c === "Neutral" ? C.t2 : C.g;
  const catBg = (c) => c === "Crisis" ? "rgba(239,68,68,0.12)" : c === "Negative" ? "rgba(245,158,11,0.12)" : c === "Neutral" ? "rgba(107,114,128,0.08)" : "rgba(16,185,129,0.12)";

  return (
    <div style={{ fontFamily: "'JetBrains Mono','SF Mono',monospace", color: C.t1, background: C.bg, minHeight: "100vh" }}>
      {toast && <div style={{ position: "fixed", top: 16, right: 16, zIndex: 999, padding: "10px 18px", borderRadius: 6, background: "rgba(16,185,129,0.15)", border: "1px solid rgba(16,185,129,0.3)", color: C.g, fontSize: 12, fontFamily: "inherit" }}>{toast}</div>}

      {/* HEADER */}
      <div style={{ borderBottom: "1px solid " + C.border, padding: "14px 24px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: C.g, boxShadow: "0 0 6px " + C.g }} />
          <span style={{ fontSize: 14, fontWeight: 600, letterSpacing: "0.06em" }}>REPUVIEW</span>
          <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: dataSource === "live" ? "rgba(16,185,129,0.12)" : "rgba(245,158,11,0.12)", color: dataSource === "live" ? C.g : C.a, fontWeight: 600, marginLeft: 8 }}>{dataSource === "live" ? "LIVE DATA" : dataSource === "demo" ? "DEMO" : "..."}</span>
        </div>
        <form onSubmit={goCustom} style={{ display: "flex", gap: 4 }}>
          <input type="text" placeholder="Search any company..." value={customName} onChange={e => setCustomName(e.target.value)}
            style={{ padding: "6px 10px", borderRadius: 5, border: "1px solid " + C.border, background: "rgba(255,255,255,0.03)", color: C.t1, fontSize: 11, fontFamily: "inherit", width: 180, outline: "none" }} />
          <button type="submit" style={{ padding: "6px 10px", borderRadius: 5, border: "1px solid rgba(16,185,129,0.25)", background: "rgba(16,185,129,0.06)", color: C.g, fontSize: 10, fontFamily: "inherit", cursor: "pointer", fontWeight: 600 }}>SEARCH</button>
        </form>
      </div>

      <div style={{ padding: "16px 24px", maxWidth: 1200, margin: "0 auto" }}>
        {/* TABS */}
        <div style={{ display: "flex", marginBottom: 20, borderBottom: "1px solid " + C.border }}>
          {["overview", "company", "news"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{ padding: "9px 18px", border: "none", borderBottom: "2px solid " + (tab === t ? C.g : "transparent"), background: "transparent", color: tab === t ? C.t1 : C.t3, fontSize: 10, fontFamily: "inherit", cursor: "pointer", letterSpacing: "0.08em", fontWeight: tab === t ? 600 : 400 }}>
              {t === "news" ? "LIVE NEWS" : t === "company" ? (company || "COMPANY") : "OVERVIEW"}
            </button>
          ))}
        </div>

        {/* ═══ OVERVIEW ═══ */}
        {tab === "overview" && (<div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <span style={{ fontSize: 10, color: C.t3, letterSpacing: "0.08em" }}>MY PORTFOLIO · {portfolio.length}</span>
            <button onClick={() => setEditMode(!editMode)} style={{ padding: "4px 12px", borderRadius: 4, border: "1px solid " + (editMode ? C.a : C.border), background: editMode ? "rgba(245,158,11,0.08)" : "transparent", color: editMode ? C.a : C.t2, fontSize: 10, fontFamily: "inherit", cursor: "pointer" }}>{editMode ? "DONE" : "EDIT"}</button>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(210px,1fr))", gap: 10, marginBottom: 20 }}>
            {portfolio.map(p => { const d = PRE[p.ticker]; const l = d?.data[d.data.length - 1]; const sp = d?.data.slice(-30) || []; const mn = sp.length ? Math.min(...sp.map(x => x.index)) - 0.5 : 99; const mx = sp.length ? Math.max(...sp.map(x => x.index)) + 0.5 : 101; const col = TCOL[p.ticker] || C.b;
              return (<div key={p.ticker} style={{ position: "relative", padding: 14, borderRadius: 8, border: "1px solid " + C.border, background: C.card, cursor: editMode ? "default" : "pointer" }}
                onClick={() => { if (!editMode) { pick(p.ticker, p.name); setTab("company"); } }}>
                {editMode && <button onClick={e => { e.stopPropagation(); rmPort(p.ticker); }} style={{ position: "absolute", top: 6, right: 6, width: 20, height: 20, borderRadius: "50%", border: "1px solid rgba(239,68,68,0.3)", background: "rgba(239,68,68,0.1)", color: C.r, fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "inherit" }}>×</button>}
                <div style={{ marginBottom: 6 }}><span style={{ fontSize: 13, fontWeight: 600 }}>{p.ticker}</span><span style={{ fontSize: 10, color: C.t3, marginLeft: 6 }}>{p.name}</span></div>
                {l ? (<>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 2 }}>
                    <span style={{ fontSize: 24, fontWeight: 700 }}>{fi(l.index)}</span>
                    <span style={{ fontSize: 12, color: cc(l.weeklyChg), fontWeight: 600 }}>{fc(l.weeklyChg)}</span>
                  </div>
                  <div style={{ fontSize: 10, color: C.t3, marginBottom: 8 }}>vs MA: <span style={{ color: cc(l.devFromMA) }}>{fc(l.devFromMA)}</span> · 4w: <span style={{ color: cc(l.monthlyChg) }}>{fc(l.monthlyChg)}</span></div>
                  <div style={{ height: 52 }}><ResponsiveContainer width="100%" height="100%"><ComposedChart data={sp}><YAxis domain={[mn, mx]} hide /><Area type="monotone" dataKey="index" stroke={col} fill={col} fillOpacity={0.12} strokeWidth={2} dot={false} /><Line type="monotone" dataKey="ma30" stroke="rgba(255,255,255,0.12)" strokeWidth={1} strokeDasharray="2 2" dot={false} /></ComposedChart></ResponsiveContainer></div>
                </>) : <div style={{ fontSize: 11, color: C.t3, padding: "16px 0" }}>Custom — use LIVE NEWS + AI</div>}
              </div>); })}
            {editMode && (<div style={{ padding: 14, borderRadius: 8, border: "1px dashed rgba(16,185,129,0.25)", background: "rgba(16,185,129,0.02)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 140, gap: 8 }}>
              <div style={{ fontSize: 10, color: C.t3 }}>Add company</div>
              <form onSubmit={e => { e.preventDefault(); const v = e.target.elements.at.value.trim(); if (v) { addPort(v.toUpperCase().replace(/\s+/g, "").slice(0, 5), v); e.target.elements.at.value = ""; } }} style={{ display: "flex", gap: 4 }}>
                <input name="at" type="text" placeholder="Name..." style={{ padding: "6px 8px", borderRadius: 4, border: "1px solid " + C.border, background: "rgba(255,255,255,0.03)", color: C.t1, fontSize: 11, fontFamily: "inherit", width: 120, outline: "none" }} />
                <button type="submit" style={{ padding: "6px 8px", borderRadius: 4, border: "1px solid rgba(16,185,129,0.3)", background: "rgba(16,185,129,0.08)", color: C.g, fontSize: 11, fontFamily: "inherit", cursor: "pointer" }}>+</button>
              </form>
            </div>)}
          </div>
          {/* Spillover */}
          <div style={{ padding: 16, borderRadius: 8, border: "1px solid " + C.border, background: C.card, marginBottom: 16 }}>
            <div style={{ fontSize: 10, color: C.t3, letterSpacing: "0.08em", marginBottom: 14 }}>REPUTATION CONTAGION · SPILLOVER INDEX: {spillTSI}%</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(100px,1fr))", gap: 8 }}>
              {SPILL.map(s => (<div key={s.t} style={{ textAlign: "center", padding: "10px 6px", borderRadius: 6, background: s.v > 0 ? "rgba(239,68,68,0.04)" : "rgba(16,185,129,0.04)", border: "1px solid " + (s.v > 0 ? "rgba(239,68,68,0.1)" : "rgba(16,185,129,0.1)") }}>
                <div style={{ fontSize: 12, fontWeight: 600 }}>{s.t}</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: s.v > 0 ? C.r : C.g, marginTop: 2 }}>{s.v > 0 ? "+" : ""}{s.v}%</div>
                <div style={{ fontSize: 8, color: C.t3, letterSpacing: "0.1em", marginTop: 2 }}>{s.v > 0 ? "TRANSMITTER" : "RECEIVER"}</div>
              </div>))}
            </div>
          </div>
          {/* All-co chart */}
          <div style={{ padding: 16, borderRadius: 8, border: "1px solid " + C.border, background: C.card }}>
            <div style={{ fontSize: 10, color: C.t3, letterSpacing: "0.08em", marginBottom: 12 }}>ALL COMPANIES · WEEKLY INDEX</div>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart><CartesianGrid stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="date" tick={{ fontSize: 9, fill: C.t3 }} tickLine={false} axisLine={false} type="category" allowDuplicatedCategory={false} tickFormatter={v => { const d = new Date(v); return d.getMonth() === 0 ? String(d.getFullYear()) : ""; }} />
                <YAxis domain={[88, 112]} tick={{ fontSize: 9, fill: C.t3 }} tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTip />} />
                {Object.entries(PRE).map(([t, info]) => (<Line key={t} data={info.data.filter((_, i) => i % 2 === 0)} dataKey="index" stroke={TCOL[t]} strokeWidth={1.5} dot={false} name={t} opacity={0.85} />))}
              </LineChart>
            </ResponsiveContainer>
            <div style={{ display: "flex", gap: 14, justifyContent: "center", marginTop: 8 }}>{Object.entries(TCOL).map(([k, c]) => (<div key={k} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: C.t2 }}><div style={{ width: 12, height: 3, background: c, borderRadius: 1 }} />{k}</div>))}</div>
          </div>
        </div>)}

        {/* ═══ COMPANY TAB ═══ */}
        {tab === "company" && hist && (<div>
          {/* Profile card */}
          {curProfile && (<div style={{ padding: "16px 20px", borderRadius: 8, border: "1px solid " + C.border, background: C.card, marginBottom: 16, display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
            <LogoImg domain={curProfile.domain} name={curProfile.name || company} size={48} />
            <div style={{ flex: "1 1 200px", minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 4, flexWrap: "wrap" }}>
                <span style={{ fontSize: 18, fontWeight: 700 }}>{curProfile.name || hist.name}</span>
                <span style={{ fontSize: 11, color: C.t3 }}>{sel}</span>
                {curProfile.marketCap && <span style={{ fontSize: 11, color: C.b, fontWeight: 600 }}>{curProfile.marketCap}</span>}
              </div>
              <div style={{ fontSize: 11, color: C.t2, lineHeight: 1.5, marginBottom: 8 }}>{curProfile.summary}</div>
              <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 3, background: "rgba(96,165,250,0.1)", color: C.b, fontWeight: 600 }}>{curProfile.industry}</span>
                {curProfile.keywords?.map((k, i) => <span key={i} style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(255,255,255,0.04)", color: C.t3 }}>{k}</span>)}
              </div>
            </div>
            {latest && <div style={{ textAlign: "right", flexShrink: 0 }}>
              <div style={{ fontSize: 28, fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>{fi(latest.index)}</div>
              <div style={{ fontSize: 11 }}><span style={{ color: cc(latest.weeklyChg), fontWeight: 600 }}>{fc(latest.weeklyChg)} 1w</span> · <span style={{ color: cc(latest.monthlyChg) }}>{fc(latest.monthlyChg)} 4w</span></div>
              <div style={{ fontSize: 10, color: C.t3 }}>vs MA: <span style={{ color: cc(latest.devFromMA) }}>{fc(latest.devFromMA)}</span></div>
            </div>}
          </div>)}
          <div style={{ padding: 16, borderRadius: 8, border: "1px solid " + C.border, background: C.card, marginBottom: 12 }}>
            <div style={{ fontSize: 10, color: C.t3, letterSpacing: "0.08em", marginBottom: 10 }}>REPUTATION INDEX · WEEKLY · 30W MA</div>
            <ResponsiveContainer width="100%" height={300}><ComposedChart data={hist.data}><CartesianGrid stroke="rgba(255,255,255,0.03)" /><XAxis dataKey="date" tick={{ fontSize: 9, fill: C.t3 }} tickLine={false} axisLine={false} tickFormatter={v => { const d = new Date(v); return d.getMonth() % 6 === 0 ? d.toLocaleDateString("en-US", { month: "short", year: "2-digit" }) : ""; }} /><YAxis domain={["auto", "auto"]} tick={{ fontSize: 9, fill: C.t3 }} tickLine={false} axisLine={false} /><Tooltip contentStyle={{ background: "#1a1a1e", border: "1px solid " + C.border, borderRadius: 6, fontSize: 11 }} formatter={(v, n) => [fi(v), n === "index" ? "Index" : "30w MA"]} /><ReferenceLine y={100} stroke="rgba(255,255,255,0.08)" strokeDasharray="4 4" /><defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={C.g} stopOpacity={0.15} /><stop offset="100%" stopColor={C.g} stopOpacity={0} /></linearGradient></defs><Area type="monotone" dataKey="index" stroke={C.g} fill="url(#sg)" strokeWidth={2} dot={false} /><Line type="monotone" dataKey="ma30" stroke={C.a} strokeWidth={1} strokeDasharray="3 3" dot={false} /></ComposedChart></ResponsiveContainer>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div style={{ padding: 16, borderRadius: 8, border: "1px solid " + C.border, background: C.card }}><div style={{ fontSize: 10, color: C.t3, letterSpacing: "0.08em", marginBottom: 10 }}>% DEVIATION FROM MA</div><ResponsiveContainer width="100%" height={110}><BarChart data={hist.data.filter((_, i) => i % 2 === 0)}><XAxis dataKey="date" tick={false} axisLine={false} /><YAxis tick={{ fontSize: 9, fill: C.t3 }} tickLine={false} axisLine={false} /><ReferenceLine y={0} stroke="rgba(255,255,255,0.08)" /><Bar dataKey="devFromMA" radius={[1, 1, 0, 0]} shape={props => <rect x={props.x} y={props.y} width={props.width} height={Math.abs(props.height)} fill={props.value > 0 ? C.g : C.r} opacity={0.5} rx={1} />} /></BarChart></ResponsiveContainer></div>
            <div style={{ padding: 16, borderRadius: 8, border: "1px solid " + C.border, background: C.card }}><div style={{ fontSize: 10, color: C.t3, letterSpacing: "0.08em", marginBottom: 10 }}>20W VOLATILITY</div><ResponsiveContainer width="100%" height={110}><AreaChart data={hist.data.filter(d => d.vol20w != null)}><XAxis dataKey="date" tick={false} axisLine={false} /><YAxis tick={{ fontSize: 9, fill: C.t3 }} tickLine={false} axisLine={false} /><Area type="monotone" dataKey="vol20w" stroke={C.p} fill={C.p} fillOpacity={0.1} strokeWidth={1.2} dot={false} /></AreaChart></ResponsiveContainer></div>
          </div>
        </div>)}

        {/* Company tab: custom company */}
        {tab === "company" && isCustom && (
          <div>
            {curProfile ? (<div style={{ padding: "16px 20px", borderRadius: 8, border: "1px solid " + C.border, background: C.card, marginBottom: 16, display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
              <LogoImg domain={curProfile.domain} name={curProfile.name || company} size={48} />
              <div style={{ flex: "1 1 200px", minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 4, flexWrap: "wrap" }}>
                  <span style={{ fontSize: 18, fontWeight: 700 }}>{curProfile.name || company}</span>
                  {curProfile.marketCap && <span style={{ fontSize: 11, color: C.b, fontWeight: 600 }}>{curProfile.marketCap}</span>}
                </div>
                <div style={{ fontSize: 11, color: C.t2, lineHeight: 1.5, marginBottom: 8 }}>{curProfile.summary}</div>
                <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                  <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 3, background: "rgba(96,165,250,0.1)", color: C.b, fontWeight: 600 }}>{curProfile.industry}</span>
                  {curProfile.keywords?.map((k, i) => <span key={i} style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "rgba(255,255,255,0.04)", color: C.t3 }}>{k}</span>)}
                </div>
              </div>
            </div>) : (<div style={{ padding: 20, borderRadius: 8, border: "1px solid " + C.border, background: C.card, marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 14, height: 14, border: "2px solid rgba(96,165,250,0.2)", borderTopColor: C.b, borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
              <span style={{ fontSize: 11, color: C.t3 }}>Loading company profile...</span>
            </div>)}
          </div>
        )}

        {/* Company tab: shared projection + insights (shows for both preloaded and custom) */}
        {tab === "company" && (<div>
          {/* Progress */}
          {newsLoading && <div style={{ textAlign: "center", padding: 30 }}><div style={{ width: 20, height: 20, border: "2px solid rgba(16,185,129,0.15)", borderTopColor: C.g, borderRadius: "50%", animation: "spin 0.8s linear infinite", margin: "0 auto" }} /><div style={{ fontSize: 11, color: C.t3, marginTop: 10 }}>Fetching latest news for analysis...</div></div>}
          {news.length > 0 && !allDone && analyzed.length > 0 && (<div style={{ marginTop: 12, marginBottom: 12, padding: "10px 16px", borderRadius: 6, background: "rgba(167,139,250,0.04)", border: "1px solid rgba(167,139,250,0.1)", display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 10, height: 10, border: "1.5px solid rgba(167,139,250,0.2)", borderTopColor: C.p, borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
            <span style={{ fontSize: 11, color: C.p }}>Analyzing {analyzed.length}/{news.length} articles · Forecast + insights appear when complete</span>
          </div>)}

          {/* DECAY PROJECTION */}
          {proj.length > 0 && (<div style={{ marginTop: 16, padding: 16, borderRadius: 8, border: "1px solid " + C.border, background: C.card }}>
            <div style={{ fontSize: 10, color: C.t3, letterSpacing: "0.08em", marginBottom: 4 }}>REPUTATION IMPACT FORECAST · 12-WEEK PROJECTION</div>
            <div style={{ fontSize: 11, color: C.t2, marginBottom: 12 }}>Based on decay curves of {analyzed.length} active events.</div>
            <ResponsiveContainer width="100%" height={180}><ComposedChart data={proj.filter((_, i) => i % 2 === 0)}><CartesianGrid stroke="rgba(255,255,255,0.03)" /><XAxis dataKey="day" tick={{ fontSize: 9, fill: C.t3 }} tickLine={false} axisLine={false} tickFormatter={v => v % 14 === 0 ? "W" + v / 7 : ""} /><YAxis tick={{ fontSize: 9, fill: C.t3 }} tickLine={false} axisLine={false} /><Tooltip contentStyle={{ background: "#1a1a1e", border: "1px solid " + C.border, borderRadius: 6, fontSize: 11 }} formatter={(v, n) => [v.toFixed(2), n === "net" ? "Net" : n]} labelFormatter={v => "Day " + v + " (W" + Math.floor(v / 7) + ")"} /><ReferenceLine y={0} stroke="rgba(255,255,255,0.08)" strokeDasharray="4 4" /><Area type="monotone" dataKey="positive" stroke={C.g} fill={C.g} fillOpacity={0.08} strokeWidth={1} dot={false} /><Area type="monotone" dataKey="negative" stroke={C.r} fill={C.r} fillOpacity={0.08} strokeWidth={1} dot={false} /><Line type="monotone" dataKey="net" stroke={C.b} strokeWidth={2} dot={false} /></ComposedChart></ResponsiveContainer>
            <div style={{ display: "flex", gap: 16, marginTop: 12, fontSize: 10 }}>
              {[["NOW", 0], ["W4", 28], ["W8", 56], ["W12", 84]].map(([label, idx]) => (<div key={label} style={{ padding: "8px 12px", borderRadius: 6, background: "rgba(255,255,255,0.02)", border: "1px solid " + C.border, flex: 1, textAlign: "center" }}>
                <div style={{ color: C.t3, marginBottom: 2 }}>{label}</div>
                <div style={{ fontWeight: 700, color: impColor(proj[idx]?.net || 0) }}>{(proj[idx]?.net || 0) >= 0 ? "+" : ""}{(proj[idx]?.net || 0).toFixed(2)}</div>
              </div>))}
            </div>
          </div>)}

          {/* AI INSIGHTS — SWOT + Recovery Plan */}
          {(insightsLoading || insights) && (<div style={{ marginTop: 16, padding: 16, borderRadius: 8, border: "1px solid rgba(96,165,250,0.15)", background: "rgba(96,165,250,0.03)" }}>
            <div style={{ fontSize: 10, color: C.b, letterSpacing: "0.08em", fontWeight: 600, marginBottom: 12 }}>AI STRATEGIC INSIGHTS · SWOT + RECOVERY PLAN</div>
            {insightsLoading && <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "20px 0" }}><div style={{ width: 14, height: 14, border: "2px solid rgba(96,165,250,0.2)", borderTopColor: C.b, borderRadius: "50%", animation: "spin 0.8s linear infinite" }} /><span style={{ fontSize: 11, color: C.t3 }}>Generating SWOT analysis and recovery plan...</span></div>}
            {insights && (<div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

              {/* Trajectory */}
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 600 }}>Trajectory</span>
                  <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 3, fontWeight: 600, background: insights.trajectory === "improving" ? "rgba(16,185,129,0.12)" : insights.trajectory === "deteriorating" ? "rgba(239,68,68,0.12)" : insights.trajectory === "volatile" ? "rgba(245,158,11,0.12)" : "rgba(107,114,128,0.08)", color: insights.trajectory === "improving" ? C.g : insights.trajectory === "deteriorating" ? C.r : insights.trajectory === "volatile" ? C.a : C.t2 }}>{(insights.trajectory || "").toUpperCase()}</span>
                </div>
                <div style={{ fontSize: 11, color: C.t2, lineHeight: 1.5 }}>{insights.trajectory_summary}</div>
              </div>

              {/* SWOT Grid */}
              {insights.swot && (<div>
                <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 10 }}>SWOT Analysis</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  {[
                    { key: "strengths", label: "STRENGTHS", color: C.g, bg: "rgba(16,185,129,0.06)", border: "rgba(16,185,129,0.15)" },
                    { key: "weaknesses", label: "WEAKNESSES", color: C.r, bg: "rgba(239,68,68,0.06)", border: "rgba(239,68,68,0.15)" },
                    { key: "opportunities", label: "OPPORTUNITIES", color: C.b, bg: "rgba(96,165,250,0.06)", border: "rgba(96,165,250,0.15)" },
                    { key: "threats", label: "THREATS", color: C.a, bg: "rgba(245,158,11,0.06)", border: "rgba(245,158,11,0.15)" },
                  ].map(q => (<div key={q.key} style={{ padding: "10px 12px", borderRadius: 6, background: q.bg, border: "1px solid " + q.border }}>
                    <div style={{ fontSize: 9, color: q.color, fontWeight: 600, letterSpacing: "0.08em", marginBottom: 6 }}>{q.label}</div>
                    {(insights.swot[q.key] || []).map((item, idx) => (
                      <div key={idx} style={{ fontSize: 10, color: C.t2, lineHeight: 1.5, marginBottom: 3, paddingLeft: 8, borderLeft: "2px solid " + q.border }}>
                        {item}
                      </div>
                    ))}
                  </div>))}
                </div>
              </div>)}

              {/* Reputation moat */}
              {insights.reputation_moat && <div style={{ padding: "10px 14px", borderRadius: 6, background: "rgba(16,185,129,0.04)", border: "1px solid rgba(16,185,129,0.1)" }}><div style={{ fontSize: 9, color: C.g, fontWeight: 600, marginBottom: 4 }}>REPUTATION MOAT — CORE ASSET</div><div style={{ fontSize: 11, color: C.t2, lineHeight: 1.5 }}>{insights.reputation_moat}</div></div>}

              {/* Recovery Plan — 3 horizons */}
              {insights.recovery_plan && (<div>
                <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 10 }}>Reputation Recovery / Growth Plan</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {[
                    { key: "short_term", label: "SHORT-TERM", color: C.r, bg: "rgba(239,68,68,0.04)", border: "rgba(239,68,68,0.12)" },
                    { key: "medium_term", label: "MEDIUM-TERM", color: C.a, bg: "rgba(245,158,11,0.04)", border: "rgba(245,158,11,0.12)" },
                    { key: "long_term", label: "LONG-TERM", color: C.g, bg: "rgba(16,185,129,0.04)", border: "rgba(16,185,129,0.12)" },
                  ].map(h => {
                    const plan = insights.recovery_plan[h.key];
                    if (!plan) return null;
                    return (<div key={h.key} style={{ padding: "14px 16px", borderRadius: 8, background: h.bg, border: "1px solid " + h.border }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                        <span style={{ fontSize: 10, color: h.color, fontWeight: 600, letterSpacing: "0.08em" }}>{h.label}</span>
                        <span style={{ fontSize: 9, color: C.t3 }}>{plan.timeframe}</span>
                      </div>
                      {plan.key_events && <div style={{ fontSize: 10, color: C.t3, marginBottom: 8, fontStyle: "italic" }}>Driven by: {plan.key_events}</div>}
                      <div style={{ fontSize: 11, color: C.t2, lineHeight: 1.5, marginBottom: 8 }}>{plan.messaging}</div>
                      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 6, fontSize: 10 }}>
                        <div><span style={{ color: C.t3 }}>Target: </span><span style={{ color: C.t1, fontWeight: 600 }}>{plan.target_audience}</span></div>
                      </div>
                      {plan.channels?.length > 0 && <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 8 }}>
                        {plan.channels.map((ch, j) => <span key={j} style={{ fontSize: 8, padding: "2px 6px", borderRadius: 3, background: "rgba(167,139,250,0.08)", color: C.p }}>{ch}</span>)}
                      </div>}
                      {plan.actions?.length > 0 && <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                        {plan.actions.map((a, j) => <div key={j} style={{ fontSize: 10, color: C.t2, paddingLeft: 8, borderLeft: "2px solid " + h.border }}>{a}</div>)}
                      </div>}
                    </div>);
                  })}
                </div>
              </div>)}

            </div>)}
          </div>)}
        </div>)}

        {/* ═══ NEWS + AI ═══ */}
        {tab === "news" && (<div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontSize: 14, fontWeight: 600 }}>Live reputation intelligence</span>
              <span style={{ fontSize: 11, color: C.t3 }}>— {company}</span>
              {!portfolio.find(p => p.name.toLowerCase() === company.toLowerCase()) && <button onClick={() => addPort(company.toUpperCase().replace(/\s+/g, "").slice(0, 5), company)} style={{ padding: "4px 10px", borderRadius: 4, border: "1px solid rgba(16,185,129,0.3)", background: "rgba(16,185,129,0.06)", color: C.g, fontSize: 10, fontFamily: "inherit", cursor: "pointer", fontWeight: 600 }}>+ ADD TO OVERVIEW</button>}
            </div>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              {analyzed.length > 0 && <span style={{ fontSize: 11, color: C.t2 }}>Net: <span style={{ color: impColor(totalRes), fontWeight: 600 }}>{totalRes >= 0 ? "+" : ""}{totalRes.toFixed(2)}</span></span>}
              <button onClick={() => { nRef.current = {}; iRef.current = {}; setNews([]); setInsights(null); fetchNews(company); }} style={{ padding: "5px 12px", borderRadius: 5, border: "1px solid " + C.border, background: "transparent", color: C.t2, fontSize: 10, fontFamily: "inherit", cursor: "pointer" }}>REFRESH</button>
            </div>
          </div>

          {newsLoading && <div style={{ textAlign: "center", padding: 50 }}><div style={{ width: 20, height: 20, border: "2px solid rgba(16,185,129,0.15)", borderTopColor: C.g, borderRadius: "50%", animation: "spin 0.8s linear infinite", margin: "0 auto" }} /><div style={{ fontSize: 11, color: C.t3, marginTop: 10 }}>Searching...</div></div>}

          {/* News items */}
          {news.length > 0 && !newsLoading && (<div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {news.map((item, i) => (<div key={i} style={{ padding: "14px 16px", borderRadius: 8, border: "1px solid " + C.border, background: C.card, display: "flex", gap: 14, flexWrap: "wrap" }}>
              <div style={{ flex: "1 1 280px", minWidth: 0 }}>
                <div style={{ fontSize: 13, fontWeight: 600, lineHeight: 1.4, marginBottom: 5 }}>{item.headline}</div>
                <div style={{ fontSize: 10, color: C.t3, marginBottom: 5 }}>{item.source}{item.date ? " · " + item.date : ""}{item.days_ago != null ? " · " + item.days_ago + "d ago" : ""}</div>
                <div style={{ fontSize: 11, color: C.t2, lineHeight: 1.5 }}>{item.summary}</div>
              </div>
              <div style={{ flex: "0 0 240px", padding: "10px 14px", borderRadius: 6, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.03)" }}>
                <div style={{ fontSize: 9, color: C.t3, letterSpacing: "0.1em", marginBottom: 8 }}>AI REPUTATION IMPACT</div>
                {item.loading && <div style={{ display: "flex", alignItems: "center", gap: 6 }}><div style={{ width: 10, height: 10, border: "1.5px solid rgba(167,139,250,0.2)", borderTopColor: C.p, borderRadius: "50%", animation: "spin 0.8s linear infinite" }} /><span style={{ fontSize: 10, color: C.t3 }}>Analyzing...</span></div>}
                {item.impact && !item.loading && (<>
                  <div style={{ display: "flex", gap: 12, marginBottom: 8 }}>
                    <div><div style={{ fontSize: 8, color: C.t3, marginBottom: 2 }}>PEAK</div><div style={{ fontSize: 18, fontWeight: 700, color: impColor(item.impact.peak_impact) }}>{item.impact.peak_impact > 0 ? "+" : ""}{item.impact.peak_impact.toFixed(1)}</div></div>
                    <div style={{ width: 1, background: C.border }} />
                    <div><div style={{ fontSize: 8, color: C.t3, marginBottom: 2 }}>CURRENT</div><div style={{ fontSize: 18, fontWeight: 700, color: impColor(item.impact.residual) }}>{item.impact.residual > 0 ? "+" : ""}{item.impact.residual.toFixed(2)}</div></div>
                    <div style={{ width: 1, background: C.border }} />
                    <div><div style={{ fontSize: 8, color: C.t3, marginBottom: 2 }}>DECAYED</div><div style={{ fontSize: 14, fontWeight: 600, color: C.t2, marginTop: 2 }}>{item.impact.decay_pct}%</div></div>
                  </div>
                  <div style={{ display: "flex", gap: 6, marginBottom: 6, flexWrap: "wrap" }}>
                    <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 3, fontWeight: 600, background: catBg(item.impact.category), color: catColor(item.impact.category) }}>{item.impact.category}</span>
                    <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 3, background: "rgba(167,139,250,0.08)", color: C.p }}>{item.impact.duration} · t½={item.impact.halflife_days}d</span>
                  </div>
                  <div style={{ fontSize: 10, color: C.t2, lineHeight: 1.4, marginBottom: 5 }}>{item.impact.reasoning}</div>
                  {item.impact.stakeholders?.length > 0 && <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>{item.impact.stakeholders.map((s, j) => <span key={j} style={{ fontSize: 8, padding: "1px 5px", borderRadius: 2, background: "rgba(255,255,255,0.04)", color: C.t3 }}>{s}</span>)}</div>}
                </>)}
              </div>
            </div>))}
          </div>)}

          {/* Model note */}
          <div style={{ marginTop: 16, padding: 14, borderRadius: 8, background: "rgba(167,139,250,0.03)", border: "1px solid rgba(167,139,250,0.08)" }}>
            <div style={{ fontSize: 10, color: C.p, fontWeight: 600, marginBottom: 4 }}>Decay model</div>
            <div style={{ fontSize: 10, color: C.t3, lineHeight: 1.6 }}>residual = peak × e^(-t/t½) · Half-lives: short=7d, medium=28d, long=84d · Scale: ±5=existential, ±3=major, ±1=notable, ±0.5=minor</div>
          </div>
        </div>)}
      </div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}@keyframes fadeIn{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:translateY(0)}}*{box-sizing:border-box;margin:0}button:hover{filter:brightness(1.15)}input:focus{border-color:rgba(16,185,129,0.35)!important}`}</style>
    </div>
  );
}
