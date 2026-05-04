"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Logo, Sparkline, BarChart, Donut, Stat, Avatar, Toggle } from "./primitives";
import { Icon } from "./icons";
import { useI18n, LangSwitcher } from "./i18n-provider";
import { ThemeToggle } from "./theme-provider";
import { useToast } from "./toast";
import { useAuth } from "./auth-provider";

type Pane = "overview" | "attacks" | "models" | "billing" | "integrations";

// ── Sidebar ────────────────────────────────────────────────────
function DashSidebar({ active, setActive }: { active: Pane; setActive: (p: Pane) => void }) {
  const { t } = useI18n();
  const items: { id: Pane; label: string; icon: React.FC<{ size?: number }> }[] = [
    { id: "overview", label: t.dPaneOverview, icon: Icon.layers },
    { id: "attacks", label: t.dPaneAttacks, icon: Icon.warn },
    { id: "models", label: t.dPaneModels, icon: Icon.cpu },
    { id: "billing", label: t.dPaneBilling, icon: Icon.spark },
    { id: "integrations", label: t.dPaneIntegrations, icon: Icon.shield },
  ];
  return (
    <div style={{ width: 220, borderRight: "1px solid var(--line)", background: "var(--surface)", display: "flex", flexDirection: "column", padding: "20px 12px", gap: 4, flexShrink: 0 }}>
      <div style={{ padding: "6px 8px 14px" }}>
        <Link href="/" style={{ textDecoration: "none", color: "inherit" }}>
          <Logo size={13} />
        </Link>
      </div>
      {items.map((item) => {
        const I = item.icon;
        const isActive = active === item.id;
        return (
          <button
            key={item.id}
            onClick={() => setActive(item.id)}
            style={{
              appearance: "none", border: 0, cursor: "pointer",
              display: "flex", alignItems: "center", gap: 10, padding: "8px 10px",
              borderRadius: 7, background: isActive ? "var(--bg-2)" : "transparent",
              color: isActive ? "var(--ink)" : "var(--muted)",
              fontSize: 13, fontWeight: isActive ? 500 : 400, fontFamily: "var(--font-sans)",
              transition: "background 0.1s", textAlign: "left",
            }}
          >
            <I size={15} /> <span>{item.label}</span>
          </button>
        );
      })}
      <div style={{ flex: 1 }} />
      <div style={{ padding: 12, background: "var(--bg-2)", borderRadius: 8, fontSize: 11.5 }}>
        <div style={{ color: "var(--muted)", marginBottom: 4 }}>{t.dBalance}</div>
        <div className="mono" style={{ fontSize: 16, fontWeight: 500, color: "var(--ink)" }}>₸ 184,200</div>
        <div style={{ marginTop: 6, color: "var(--muted-2)", fontSize: 10.5 }}>{t.dDaysLeft}</div>
      </div>
    </div>
  );
}

// ── Topbar ─────────────────────────────────────────────────────
const NOTIFS = [
  { title: "Attack spike on examplebank.kz", desc: "+34% in last hour", time: "2m ago", dot: "var(--danger)", email: "admin@examplebank.kz" },
  { title: "vrf_8fa4b21c verified", desc: "Confidence 99.4%", time: "5m ago", dot: "var(--ok)", email: "user@cryptopay.kz" },
  { title: "Low balance alert", desc: "≈ 14 days remaining", time: "1h ago", dot: "var(--warn)", email: "billing@medexpert.kz" },
];

function DashTopbar({ title, subtitle }: { title: string; subtitle?: string }) {
  const { t } = useI18n();
  const { logout, session } = useAuth();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [bellOpen, setBellOpen] = useState(false);
  const [read, setRead] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const bellRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
      if (bellRef.current && !bellRef.current.contains(e.target as Node)) setBellOpen(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "18px 28px", borderBottom: "1px solid var(--line)", background: "var(--surface)", flexShrink: 0 }}>
      <div>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: "-0.02em" }}>{title}</h1>
        {subtitle && <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>{subtitle}</div>}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <LangSwitcher />
        <ThemeToggle />

        {/* Notifications bell */}
        <div ref={bellRef} style={{ position: "relative" }}>
          <button
            onClick={() => setBellOpen((v) => !v)}
            className="btn btn-ghost"
            style={{ padding: "7px 10px", position: "relative" }}
          >
            <Icon.bell size={15} />
            {!read && (
              <span style={{
                position: "absolute", top: 6, right: 6,
                width: 7, height: 7, borderRadius: "50%",
                background: "var(--danger)", border: "2px solid var(--surface)",
              }} />
            )}
          </button>
          {bellOpen && (
            <div style={{
              position: "absolute", top: "calc(100% + 8px)", right: 0, zIndex: 50,
              background: "var(--surface)", border: "1px solid var(--line)",
              borderRadius: 10, minWidth: 300,
              boxShadow: "0 8px 32px rgba(0,0,0,0.15)", overflow: "hidden",
            }}>
              <div style={{ padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--line)" }}>
                <span style={{ fontSize: 13, fontWeight: 500 }}>{t.dNotifs}</span>
                <button
                  className="btn btn-ghost"
                  onClick={() => setRead(true)}
                  style={{ padding: "3px 8px", fontSize: 11 }}
                >
                  {t.dMarkRead}
                </button>
              </div>
              {NOTIFS.map((n, i) => (
                <div key={i} style={{
                  padding: "12px 16px",
                  borderBottom: i < NOTIFS.length - 1 ? "1px solid var(--line)" : "none",
                  display: "flex", alignItems: "flex-start", gap: 10,
                  background: !read ? "var(--bg)" : "var(--surface)",
                }}>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: n.dot, flexShrink: 0, marginTop: 4 }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12.5, fontWeight: 500, color: "var(--ink-2)" }}>{n.title}</div>
                    <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{n.desc}</div>
                    <div style={{ fontSize: 10.5, color: "var(--muted-2)", marginTop: 3, fontFamily: "var(--font-mono)" }}>{n.email}</div>
                  </div>
                  <span style={{ fontSize: 10.5, color: "var(--muted-2)", fontFamily: "var(--font-mono)", flexShrink: 0, marginTop: 2 }}>{n.time}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <button className="btn btn-ghost" style={{ padding: "7px 12px", fontSize: 12.5 }}>
          <Icon.refresh size={13} /> {t.dLastWeek}
        </button>
        <div ref={ref} style={{ position: "relative" }}>
          <button
            onClick={() => setOpen((v) => !v)}
            style={{ background: "none", border: 0, cursor: "pointer", padding: 0, borderRadius: "50%" }}
          >
            <Avatar name="AC" hue={245} />
          </button>
          {open && (
            <div style={{
              position: "absolute", top: "calc(100% + 8px)", right: 0, zIndex: 50,
              background: "var(--surface)", border: "1px solid var(--line)",
              borderRadius: 10, padding: 4, minWidth: 200,
              boxShadow: "0 8px 32px rgba(0,0,0,0.3)",
            }}>
              <div style={{ padding: "10px 14px 10px" }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>{session?.companyName ?? "Company"}</div>
                <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 2, fontFamily: "var(--font-mono)" }}>{session?.login ?? ""}</div>
              </div>
              <div style={{ height: 1, background: "var(--line)", margin: "0 4px" }} />
              <button
                onClick={() => { setOpen(false); logout(); router.push("/login"); }}
                style={{
                  display: "flex", alignItems: "center", gap: 8, width: "100%",
                  padding: "9px 14px", marginTop: 4,
                  background: "none", border: 0, cursor: "pointer", borderRadius: 7,
                  fontSize: 13, color: "var(--danger)", fontFamily: "var(--font-sans)",
                  transition: "background 0.1s",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "oklch(0.35 0.12 25 / 0.15)")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "none")}
              >
                <Icon.x size={14} /> {t.dLogout ?? "Log out"}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Pane: Overview ─────────────────────────────────────────────
function PaneOverview() {
  const { t } = useI18n();
  const [search, setSearch] = useState("");
  const sparkData  = [12, 18, 16, 22, 19, 28, 31, 27, 35, 32, 38, 41];
  const sparkData2 = [3, 2, 4, 1, 5, 3, 6, 4, 7, 5, 4, 6];
  const sparkData3 = [98, 99, 97, 99, 98, 96, 99, 99, 97, 98, 99, 99];
  const barData = [
    { label: "Mon", value: 142 },
    { label: "Tue", value: 168 },
    { label: "Wed", value: 192, highlight: true },
    { label: "Thu", value: 174 },
    { label: "Fri", value: 215, highlight: true },
    { label: "Sat", value: 96 },
    { label: "Sun", value: 78 },
  ];
  const stats = [
    { label: t.dTotalChecks, value: "12,847", sub: t.dTotalChecksSub, trend: "up" as const, spark: sparkData },
    { label: t.dAttacksFound, value: "342", sub: t.dAttacksFoundSub, trend: "down" as const, spark: sparkData2 },
    { label: t.dAvgTime, value: "4.7s", sub: t.dAvgTimeSub, trend: "up" as const, spark: [5,5,4.8,4.9,4.7,4.6,4.7,4.5,4.7,4.6,4.5,4.7] },
    { label: t.dAccuracy, value: "99.1%", sub: t.dAccuracySub, trend: "up" as const, spark: sparkData3 },
  ];
  const rows: [string, string, string, string, string, string, string][] = [
    ["vrf_8fa4b21c", "14:42:18", "pass",   "EFN-B7 · ViT",       "99.4%", "examplebank.kz", "ivan.petrov@examplebank.kz"],
    ["vrf_8fa4b1ab", "14:41:55", "pass",   "EFN-B7 · ViT · Liv", "98.9%", "cryptopay.kz",   "asel.nurova@cryptopay.kz"],
    ["vrf_8fa4afe2", "14:39:02", "attack", "EFN-B7 · ViT",       "12.4%", "examplebank.kz", "d.seitkali@examplebank.kz"],
    ["vrf_8fa4ad91", "14:37:44", "pass",   "EFN-B7",             "97.2%", "medexpert.kz",   "m.bekova@medexpert.kz"],
    ["vrf_8fa4ac0e", "14:35:11", "attack", "EFN-B7 · ViT · Liv", "8.1%",  "cryptopay.kz",   "user4821@cryptopay.kz"],
    ["vrf_8fa4aa44", "14:33:28", "pass",   "EFN-B7 · ViT",       "99.7%", "examplebank.kz", "n.zhaksybekov@examplebank.kz"],
  ];
  const filteredRows = rows.filter(
    (r) => !search || r.some((cell) => cell.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div style={{ padding: 28, display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
        {stats.map((s, i) => (
          <div key={i} className="card" style={{ padding: 18 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
              <Stat label={s.label} value={s.value} sub={s.sub} trend={s.trend} />
              <Sparkline data={s.spark} width={60} height={24} color={s.trend === "down" ? "var(--danger)" : "var(--accent)"} />
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: 16 }}>
        <div className="card" style={{ padding: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>{t.dCheckFlow}</div>
              <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 2 }}>{t.dCheckFlowSub}</div>
            </div>
            <div style={{ display: "flex", gap: 12, fontSize: 11 }}>
              {[[" var(--accent)", t.dAttacksLabel], ["oklch(0.55 0.14 245 / 0.18)", t.dCleanLabel]].map(([c, l]) => (
                <div key={l} style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--muted)" }}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: c }} />{l}
                </div>
              ))}
            </div>
          </div>
          <BarChart data={barData} height={140} />
        </div>

        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 16 }}>{t.dAttackTypes}</div>
          <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
            <Donut size={110} thickness={16} segments={[
              { value: 142, color: "oklch(0.55 0.14 245)" },
              { value: 86,  color: "oklch(0.65 0.14 245)" },
              { value: 64,  color: "oklch(0.75 0.10 245)" },
              { value: 50,  color: "oklch(0.85 0.06 245)" },
            ]}/>
            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8, fontSize: 11.5 }}>
              {([
                [t.dFaceSwap,   142, "oklch(0.55 0.14 245)"],
                ["Lip-sync",    86,  "oklch(0.65 0.14 245)"],
                ["Replay",      64,  "oklch(0.75 0.10 245)"],
                ["Mask/photo",  50,  "oklch(0.85 0.06 245)"],
              ] as [string, number, string][]).map(([label, v, c]) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: c }} />
                    <span style={{ color: "var(--ink-2)" }}>{label}</span>
                  </span>
                  <span className="mono" style={{ color: "var(--muted)" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div style={{ padding: "14px 20px", borderBottom: "1px solid var(--line)", display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
          <div style={{ fontSize: 13, fontWeight: 500 }}>{t.dRecentChecks}</div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
              <Icon.search size={13} style={{ position: "absolute", left: 9, color: "var(--muted)", pointerEvents: "none" }} />
              <input
                className="txt"
                placeholder={t.dSearch}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{ paddingLeft: 28, paddingTop: 5, paddingBottom: 5, fontSize: 12, height: 30, width: 190 }}
              />
            </div>
            <button className="btn btn-ghost" style={{ padding: "4px 10px", fontSize: 11.5 }}>{t.dAllBtn}</button>
          </div>
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
          <thead>
            <tr style={{ background: "var(--bg)" }}>
              {[t.dColId, t.dColTime, t.dColResult, t.dColModels, t.dColConf, t.dColSource, t.dColEmail].map((h) => (
                <th key={h} style={{ padding: "10px 16px", textAlign: "left", fontWeight: 500, fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", fontFamily: "var(--font-sans)", borderBottom: "1px solid var(--line)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredRows.length === 0 && (
              <tr>
                <td colSpan={7} style={{ padding: "24px 20px", textAlign: "center", color: "var(--muted)", fontSize: 12.5 }}>
                  No results for &ldquo;{search}&rdquo;
                </td>
              </tr>
            )}
            {filteredRows.map((row, i) => (
              <tr key={row[0]} style={{ borderBottom: i < filteredRows.length - 1 ? "1px solid var(--line)" : "none" }}>
                <td style={{ padding: "12px 16px" }}><span className="mono" style={{ color: "var(--ink-2)" }}>{row[0]}</span></td>
                <td style={{ padding: "12px 16px" }}><span className="mono" style={{ color: "var(--muted)" }}>{row[1]}</span></td>
                <td style={{ padding: "12px 16px" }}>
                  {row[2] === "pass"
                    ? <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 500, background: "var(--ok-soft)", color: "var(--ok)" }}>PASS</span>
                    : <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 500, background: "var(--danger-soft)", color: "var(--danger)" }}>ATTACK</span>}
                </td>
                <td style={{ padding: "12px 16px" }}><span className="mono" style={{ color: "var(--ink-2)", fontSize: 11.5 }}>{row[3]}</span></td>
                <td style={{ padding: "12px 16px" }}><span className="mono" style={{ color: row[2] === "pass" ? "var(--ok)" : "var(--danger)", fontWeight: 500 }}>{row[4]}</span></td>
                <td style={{ padding: "12px 16px", color: "var(--muted)" }}>{row[5]}</td>
                <td style={{ padding: "12px 16px" }}><span className="mono" style={{ color: "var(--muted-2)", fontSize: 11.5 }}>{row[6]}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Pane: Models ────────────────────────────────────────────────
type Model = { id: string; name: string; kind: string; accuracy: number; latency: number; price: number; recommended?: boolean };
const MODELS: Model[] = [
  { id: "efn-b7",      name: "EfficientNet-B7",    kind: "Spatial CNN",      accuracy: 96.4, latency: 1.2, price: 0.12, recommended: true },
  { id: "vit-l",       name: "Vision Transformer L",kind: "Spatio-temporal",  accuracy: 98.9, latency: 2.1, price: 0.34, recommended: true },
  { id: "xception",    name: "Xception",            kind: "Spatial CNN",      accuracy: 94.2, latency: 0.8, price: 0.08 },
  { id: "liveness-cnn",name: "Liveness CNN",        kind: "Anti-spoof",       accuracy: 99.1, latency: 0.6, price: 0.06, recommended: true },
  { id: "lipsync-tcn", name: "LipSync TCN",         kind: "Temporal",         accuracy: 92.8, latency: 1.5, price: 0.18 },
  { id: "cpu-fast",    name: "CPU-Fast",             kind: "Edge inference",   accuracy: 88.6, latency: 0.3, price: 0.02 },
];
const DEFAULT_ENABLED: Record<string, boolean> = { "efn-b7": true, "vit-l": true, "xception": false, "liveness-cnn": true, "lipsync-tcn": false, "cpu-fast": false };

function PaneModels() {
  const { t } = useI18n();
  const [enabled, setEnabled] = useState<Record<string, boolean>>(DEFAULT_ENABLED);
  const [extras, setExtras] = useState([true, false, false, true]);
  const enabledList = MODELS.filter((m) => enabled[m.id]);
  const totalPrice = enabledList.reduce((s, m) => s + m.price, 0);
  const avgAcc = enabledList.length ? enabledList.reduce((s, m) => s + m.accuracy, 0) / enabledList.length : 0;
  const totalLat = enabledList.reduce((s, m) => s + m.latency, 0);

  const extraDefs = [
    { label: t.dExtra0Label, desc: t.dExtra0Desc },
    { label: t.dExtra1Label, desc: t.dExtra1Desc },
    { label: t.dExtra2Label, desc: t.dExtra2Desc },
    { label: t.dExtra3Label, desc: t.dExtra3Desc },
  ];

  return (
    <div style={{ padding: 28, display: "flex", flexDirection: "column", gap: 20 }}>
      <div className="card" style={{ padding: 20, display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 24 }}>
        <Stat label={t.dModelsEnabled} value={`${enabledList.length} / ${MODELS.length}`} />
        <Stat label={t.dCostPerCheck} value={`₸ ${totalPrice.toFixed(2)}`} />
        <Stat label={t.dEnsembleAcc} value={`${avgAcc.toFixed(1)}%`} sub={t.dEnsembleAccSub} trend="up" />
        <Stat label={t.dExpLatency} value={`${totalLat.toFixed(1)}s`} sub={t.dExpLatencySub} />
      </div>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--line)" }}>
          <div style={{ fontSize: 13, fontWeight: 500 }}>{t.dModelCatalog}</div>
          <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 2 }}>{t.dModelCatalogSub}</div>
        </div>
        {MODELS.map((m, idx) => {
          const on = enabled[m.id];
          return (
            <div key={m.id} style={{ padding: "16px 20px", display: "grid", gridTemplateColumns: "36px 1.6fr 1fr 1fr 1fr 80px", gap: 16, alignItems: "center", borderBottom: idx < MODELS.length - 1 ? "1px solid var(--line)" : "none", background: on ? "var(--surface)" : "var(--bg)", opacity: on ? 1 : 0.65 }}>
              <div style={{ width: 32, height: 32, borderRadius: 6, background: on ? "var(--accent-soft)" : "var(--bg-2)", color: on ? "var(--accent-ink)" : "var(--muted-2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Icon.cpu size={16} />
              </div>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>{m.name}</span>
                  {m.recommended && <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 3, background: "var(--accent-soft)", color: "var(--accent-ink)", fontFamily: "var(--font-sans)", letterSpacing: "0.04em", textTransform: "uppercase" }}>{t.dRec}</span>}
                </div>
                <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 2 }}>{m.kind}</div>
              </div>
              {([
                [t.dColAccuracy, `${m.accuracy}%`],
                [t.dColLatency,  `${m.latency}s`],
                [t.dColPrice,    `₸ ${m.price.toFixed(2)}`],
              ] as [string, string][]).map(([label, val]) => (
                <div key={label}>
                  <div style={{ fontSize: 10.5, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.04em", fontFamily: "var(--font-sans)" }}>{label}</div>
                  <div className="mono" style={{ fontSize: 13, fontWeight: 500 }}>{val}</div>
                </div>
              ))}
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <Toggle value={on} onChange={(v) => setEnabled((e) => ({ ...e, [m.id]: v }))} />
              </div>
            </div>
          );
        })}
      </div>

      <div className="card" style={{ padding: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>{t.dExtraSettings}</div>
        <div style={{ fontSize: 11.5, color: "var(--muted)", marginBottom: 16 }}>{t.dExtraSettingsSub}</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {extraDefs.map((s, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 0", borderBottom: i < extraDefs.length - 1 ? "1px solid var(--line)" : "none" }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 500 }}>{s.label}</div>
                <div style={{ fontSize: 11.5, color: "var(--muted)" }}>{s.desc}</div>
              </div>
              <Toggle value={extras[i]} onChange={(v) => setExtras((prev) => { const n = [...prev]; n[i] = v; return n; })} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Pane: Attacks ───────────────────────────────────────────────
function PaneAttacks() {
  const { t } = useI18n();
  const series  = [22, 28, 18, 35, 42, 38, 51, 47, 56, 62, 58, 71, 68, 84];
  const series2 = [12, 14,  8, 16, 22, 18, 24, 21, 28, 32, 28, 38, 36, 44];
  const days = ["1", "5", "10", "15", "20", "25", "30"];

  return (
    <div style={{ padding: 28, display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
        {([
          [t.dVsPrev,      t.dVsPrevVal,      t.dVsPrevSub,      "up"],
          [t.dMainVector,  t.dMainVectorVal,  t.dMainVectorSub,  null],
          [t.dPeakTime,    t.dPeakTimeVal,    t.dPeakTimeSub,    null],
        ] as [string, string, string, "up"|"down"|null][]).map(([label, value, sub, trend], i) => (
          <div key={i} className="card" style={{ padding: 18 }}>
            <Stat label={label} value={value} sub={sub} trend={trend} />
          </div>
        ))}
      </div>

      <div className="card" style={{ padding: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 500 }}>{t.dAttackDynTitle}</div>
            <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 2 }}>{t.dAttackDynSub}</div>
          </div>
          <div style={{ display: "flex", gap: 12, fontSize: 11 }}>
            {[["var(--accent)", t.dAllAttempts], ["var(--danger)", t.dBlockedLabel]].map(([c, l]) => (
              <div key={l} style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--muted)" }}>
                <span style={{ width: 14, height: 2, background: c }} />{l}
              </div>
            ))}
          </div>
        </div>
        <div style={{ position: "relative", height: 180 }}>
          <Sparkline data={series}  width={760} height={180} color="var(--accent)" fill={true} />
          <div style={{ position: "absolute", inset: 0 }}>
            <Sparkline data={series2} width={760} height={180} color="var(--danger)" fill={false} />
          </div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8, fontSize: 10.5, color: "var(--muted-2)", fontFamily: "var(--font-mono)" }}>
          {days.map((d) => <span key={d}>{d}</span>)}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 14 }}>{t.dByType}</div>
          {([
            [t.dFaceSwap,    142, 41.5, "oklch(0.55 0.14 245)"],
            [t.dLipSyncType, 86,  25.1, "oklch(0.62 0.14 245)"],
            [t.dReplayType,  64,  18.7, "oklch(0.70 0.12 245)"],
            [t.dMaskType,    38,  11.1, "oklch(0.78 0.10 245)"],
            [t.dFullSynth,   12,   3.5, "oklch(0.84 0.08 245)"],
          ] as [string, number, number, string][]).map(([label, n, pct, color]) => (
            <div key={label} style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                <span style={{ color: "var(--ink-2)" }}>{label}</span>
                <span className="mono" style={{ color: "var(--muted)" }}>{n} · {pct}%</span>
              </div>
              <div style={{ height: 6, borderRadius: 3, background: "var(--bg-2)", overflow: "hidden" }}>
                <div style={{ width: `${pct}%`, height: "100%", background: color }} />
              </div>
            </div>
          ))}
        </div>

        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 14 }}>{t.dTopSources}</div>
          {([
            ["examplebank.kz", 5824, 0.42, "up"],
            ["cryptopay.kz",   3120, 1.84, "up"],
            ["medexpert.kz",   1842, 0.18, "down"],
            ["govportal.kz",   1264, 0.91,  null],
            ["edutech.kz",      797, 0.07, "down"],
          ] as [string, number, number, string|null][]).map(([host, vol, atkPct, trend]) => (
            <div key={host} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 0", borderBottom: "1px solid var(--line)" }}>
              <div>
                <div style={{ fontSize: 12.5, fontWeight: 500, color: "var(--ink-2)" }}>{host}</div>
                <div style={{ fontSize: 10.5, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>{vol.toLocaleString()} {t.dChecksUnit}</div>
              </div>
              <div className="mono" style={{ fontSize: 12, color: trend === "up" ? "var(--danger)" : trend === "down" ? "var(--ok)" : "var(--muted)", fontWeight: 500 }}>
                {atkPct}{t.dAttacksPct}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Pane: Billing ───────────────────────────────────────────────
function PaneBilling() {
  const { t } = useI18n();
  const { show } = useToast();
  const [topUp, setTopUp] = useState(50000);
  const presets = [10000, 50000, 100000, 500000];
  const spendData = [24,28,18,32,38,42,28,52,48,56,62,58,71,68,84,76,88,92,68,78,84,72,88,96,82,94,108,98,112,124];

  return (
    <div style={{ padding: 28, display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 16 }}>
        <div className="card" style={{ padding: 24 }}>
          <div style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-sans)", textTransform: "uppercase", letterSpacing: "0.06em" }}>{t.dCurrentBalance}</div>
          <div className="mono" style={{ fontSize: 44, fontWeight: 500, letterSpacing: "-0.02em", marginTop: 6 }}>₸ 184,200</div>
          <div style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 4 }}>{t.dBalanceSub}</div>
          <div style={{ marginTop: 24, padding: 16, background: "var(--bg-2)", borderRadius: 10 }}>
            <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 12 }}>{t.dMonthlySpend}</div>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 80 }}>
              {spendData.map((v, i) => (
                <div key={i} style={{ flex: 1, height: `${(v / 124) * 100}%`, background: "oklch(0.55 0.14 245 / 0.6)", borderRadius: 1, minHeight: 2 }} />
              ))}
            </div>
          </div>
        </div>

        <div className="card" style={{ padding: 24 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>{t.dTopUpTitle}</div>
          <div style={{ fontSize: 11.5, color: "var(--muted)", marginBottom: 16 }}>{t.dTopUpDesc}</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
            {presets.map((v) => (
              <button key={v} onClick={() => setTopUp(v)} style={{ appearance: "none", cursor: "pointer", padding: "10px 12px", borderRadius: 7, border: `1px solid ${topUp === v ? "var(--accent)" : "var(--line-2)"}`, background: topUp === v ? "var(--accent-soft)" : "var(--surface)", color: topUp === v ? "var(--accent-ink)" : "var(--ink-2)", fontSize: 13, fontWeight: 500, fontFamily: "var(--font-mono)" }}>
                ₸ {v.toLocaleString()}
              </button>
            ))}
          </div>
          <input className="txt mono" value={topUp.toLocaleString()} onChange={(e) => setTopUp(parseInt(e.target.value.replace(/\D/g, "") || "0"))} style={{ marginBottom: 12 }} />
          <button
            className="btn btn-accent"
            style={{ width: "100%" }}
            onClick={() => show(t.dTopUpOk, "info")}
          >
            {t.dTopUpBtn} ₸ {topUp.toLocaleString()}
          </button>
          <div style={{ marginTop: 12, fontSize: 10.5, color: "var(--muted-2)", textAlign: "center", fontFamily: "var(--font-mono)" }}>{t.dPayMethods}</div>
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--line)", fontSize: 13, fontWeight: 500 }}>{t.dHistoryTitle}</div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
          <tbody>
            {t.dHistoryRows.map((row, i) => (
              <tr key={i} style={{ borderBottom: i < t.dHistoryRows.length - 1 ? "1px solid var(--line)" : "none" }}>
                <td style={{ padding: "12px 20px", color: "var(--muted)", fontFamily: "var(--font-sans)", width: 80 }}>{row[0]}</td>
                <td style={{ padding: "12px 20px", color: "var(--ink-2)" }}>{row[1]}</td>
                <td style={{ padding: "12px 20px", textAlign: "right", fontFamily: "var(--font-mono)", fontWeight: 500, color: row[2].startsWith("+") ? "var(--ok)" : "var(--ink-2)" }}>{row[2]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Pane: Integrations ──────────────────────────────────────────
const API_KEY = "vrs_live_8fa4b21c9d3e2f1a7b6c5d4e3f2a1b0c";

function PaneIntegrations() {
  const { t } = useI18n();
  const { show } = useToast();
  const [copied, setCopied] = useState(false);

  const copyKey = () => {
    navigator.clipboard.writeText(API_KEY).catch(() => {});
    setCopied(true);
    show(t.dCopied);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{ padding: 28 }}>
      <div className="card" style={{ padding: 24 }}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>{t.dApiKey}</div>
        <div className="mono" style={{ padding: 12, background: "var(--bg-2)", borderRadius: 8, fontSize: 12, color: "var(--ink-2)", marginBottom: 16, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{API_KEY}</span>
          <button
            className="btn btn-ghost"
            style={{ padding: "4px 10px", fontSize: 11, flexShrink: 0, color: copied ? "var(--ok)" : undefined }}
            onClick={copyKey}
          >
            {copied ? <Icon.check size={12} /> : null}
            {copied ? " " + t.dCopied : t.dCopy}
          </button>
        </div>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>{t.dRedirectUrl}</div>
        <input className="txt mono" defaultValue="https://examplebank.kz/auth/verus/callback" style={{ marginBottom: 16 }} />
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>{t.dWebhook}</div>
        <input className="txt mono" defaultValue="https://api.examplebank.kz/v1/verus/hook" style={{ marginBottom: 16 }} />
        <button className="btn btn-accent" onClick={() => show(t.dSaved)}>{t.dSave}</button>
      </div>
    </div>
  );
}

// ── Dashboard shell ─────────────────────────────────────────────
export function Dashboard({ initialPane = "overview" }: { initialPane?: Pane }) {
  const { t } = useI18n();
  const [active, setActive] = useState<Pane>(initialPane);
  const meta: Record<Pane, { title: string; subtitle: string }> = {
    overview:     { title: t.dPaneOverview,     subtitle: t.dPaneOverviewSub },
    attacks:      { title: t.dPaneAttacks,      subtitle: t.dPaneAttacksSub },
    models:       { title: t.dPaneModels,       subtitle: t.dPaneModelsSub },
    billing:      { title: t.dPaneBilling,      subtitle: t.dPaneBillingSub },
    integrations: { title: t.dPaneIntegrations, subtitle: t.dPaneIntegrationsSub },
  };

  return (
    <div style={{ display: "flex", height: "100%", background: "var(--bg)", overflow: "hidden" }}>
      <DashSidebar active={active} setActive={setActive} />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, overflow: "hidden" }}>
        <DashTopbar title={meta[active].title} subtitle={meta[active].subtitle} />
        <div style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
          {active === "overview"     && <PaneOverview />}
          {active === "attacks"      && <PaneAttacks />}
          {active === "models"       && <PaneModels />}
          {active === "billing"      && <PaneBilling />}
          {active === "integrations" && <PaneIntegrations />}
        </div>
      </div>
    </div>
  );
}
