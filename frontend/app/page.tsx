"use client";

import Link from "next/link";
import { Logo } from "@/components/primitives";
import { useI18n, LangSwitcher } from "@/components/i18n-provider";
import { ThemeToggle } from "@/components/theme-provider";
import { useAuth } from "@/components/auth-provider";

export default function Home() {
  const { t } = useI18n();
  const { session } = useAuth();

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", display: "flex", flexDirection: "column" }}>
      {/* Navbar */}
      <nav
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "20px 48px",
          borderBottom: "1px solid var(--line)",
          background: "var(--surface)",
        }}
      >
        <Logo size={15} />
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <LangSwitcher />
          <ThemeToggle />
          <Link href="/verify" className="btn btn-ghost" style={{ fontSize: 13, padding: "8px 16px" }}>
            {t.navVerify}
          </Link>
          {session ? (
            <Link href="/dashboard" className="btn btn-accent" style={{ fontSize: 13, padding: "8px 16px" }}>
              {t.navDashboard}
            </Link>
          ) : (
            <>
              <Link href="/login" className="btn btn-ghost" style={{ fontSize: 13, padding: "8px 16px" }}>
                {t.navLogin}
              </Link>
              <Link href="/register" className="btn btn-primary" style={{ fontSize: 13, padding: "8px 16px" }}>
                {t.navRegister}
              </Link>
            </>
          )}
        </div>
      </nav>

      {/* Hero */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "80px 24px",
          textAlign: "center",
        }}
      >
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            padding: "4px 12px",
            borderRadius: 999,
            background: "var(--accent-soft)",
            color: "var(--accent-ink)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.05em",
            textTransform: "uppercase",
            marginBottom: 28,
          }}
        >
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent)" }} />
          {t.heroBadge}
        </div>

        <h1
          style={{
            margin: "0 0 20px",
            fontSize: "clamp(36px, 5vw, 64px)",
            fontWeight: 700,
            letterSpacing: "-0.03em",
            lineHeight: 1.1,
            color: "var(--ink)",
            maxWidth: 700,
          }}
        >
          {t.heroTitle}
        </h1>

        <p style={{ margin: "0 0 48px", fontSize: 18, color: "var(--muted)", lineHeight: 1.6, maxWidth: 520 }}>
          {t.heroDesc}
        </p>

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
          <Link href="/verify" className="btn btn-primary" style={{ fontSize: 15, padding: "13px 28px" }}>
            {t.heroCta}
          </Link>
          <Link href={session ? "/dashboard" : "/login"} className="btn btn-ghost" style={{ fontSize: 15, padding: "13px 28px" }}>
            {t.heroDashboard}
          </Link>
        </div>

        {/* Feature cards */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 20,
            marginTop: 80,
            maxWidth: 900,
            width: "100%",
          }}
        >
          {[
            { icon: "🎥", title: t.feat1Title, desc: t.feat1Desc },
            { icon: "🤖", title: t.feat2Title, desc: t.feat2Desc },
            { icon: "📊", title: t.feat3Title, desc: t.feat3Desc },
          ].map((f) => (
            <div key={f.title} className="card" style={{ padding: 24, textAlign: "left" }}>
              <div style={{ fontSize: 28, marginBottom: 14 }}>{f.icon}</div>
              <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 8, letterSpacing: "-0.01em" }}>{f.title}</div>
              <div style={{ fontSize: 13.5, color: "var(--muted)", lineHeight: 1.55 }}>{f.desc}</div>
            </div>
          ))}
        </div>

        {/* Stats */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 1,
            marginTop: 60,
            maxWidth: 700,
            width: "100%",
            background: "var(--line)",
            borderRadius: 12,
            overflow: "hidden",
          }}
        >
          {[
            ["99.1%", t.statAccuracy],
            ["4.7s", t.statAvgTime],
            ["12,847", t.statChecks],
            ["5", t.statTypes],
          ].map(([val, label]) => (
            <div key={label} style={{ padding: "20px 16px", background: "var(--surface)", textAlign: "center" }}>
              <div className="mono" style={{ fontSize: 22, fontWeight: 600, color: "var(--ink)", letterSpacing: "-0.02em" }}>
                {val}
              </div>
              <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 4 }}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <footer
        style={{
          borderTop: "1px solid var(--line)",
          padding: "20px 48px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: 12,
          color: "var(--muted-2)",
          fontFamily: "var(--font-mono)",
        }}
      >
        <span>{t.footerBrand}</span>
        <span>{t.footerSec}</span>
      </footer>
    </div>
  );
}
