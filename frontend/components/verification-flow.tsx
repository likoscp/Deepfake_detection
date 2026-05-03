"use client";

import React, { useState, useEffect, useRef } from "react";
import { Logo } from "./primitives";
import { Icon } from "./icons";
import { useI18n } from "./i18n-provider";

// ── Recording viewport ────────────────────────────────────────
type RecordPhase = "idle" | "countdown" | "recording" | "done";

function RecordingViewport({
  phase,
  recordSecs = 5,
  onComplete,
}: {
  phase: RecordPhase;
  recordSecs?: number;
  onComplete?: () => void;
}) {
  const { t } = useI18n();
  const [count, setCount] = useState(3);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (phase === "countdown") {
      setCount(3);
      const id = setInterval(() => setCount((c) => c - 1), 700);
      return () => clearInterval(id);
    }
    if (phase === "recording") {
      setElapsed(0);
      const start = Date.now();
      const id = setInterval(() => {
        const s = (Date.now() - start) / 1000;
        if (s >= recordSecs) {
          clearInterval(id);
          setElapsed(recordSecs);
          onComplete?.();
        } else {
          setElapsed(s);
        }
      }, 50);
      return () => clearInterval(id);
    }
  }, [phase, recordSecs, onComplete]);

  const guidance =
    phase === "idle"
      ? t.vIdle
      : phase === "countdown"
      ? t.vCountdown
      : phase === "recording"
      ? elapsed < 1.5
        ? t.vLookAt
        : elapsed < 3
        ? t.vTurnRight
        : t.vBlinkTimes
      : t.vRecDone;

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        aspectRatio: "3/4",
        borderRadius: 14,
        overflow: "hidden",
        background: "#0e1726",
        backgroundImage:
          "radial-gradient(ellipse at top, rgba(255,255,255,0.05), transparent 50%), repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0 2px, transparent 2px 4px)",
      }}
    >
      {/* Silhouette */}
      <svg
        viewBox="0 0 100 130"
        preserveAspectRatio="xMidYMid meet"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", opacity: 0.18 }}
      >
        <ellipse cx="50" cy="50" rx="20" ry="26" fill="#e8edf5" />
        <rect x="35" y="74" width="30" height="40" rx="14" fill="#e8edf5" />
      </svg>

      {/* Oval guide */}
      <svg
        viewBox="0 0 100 130"
        preserveAspectRatio="none"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
      >
        <ellipse
          cx="50" cy="55" rx="26" ry="34"
          fill="none"
          stroke={phase === "recording" ? "oklch(0.55 0.14 245)" : "rgba(255,255,255,0.4)"}
          strokeWidth="0.3"
          strokeDasharray="2 2"
        />
      </svg>

      {/* Corner brackets */}
      {(
        [
          { top: 12, left: 12 },
          { top: 12, right: 12 },
          { bottom: 12, left: 12 },
          { bottom: 12, right: 12 },
        ] as React.CSSProperties[]
      ).map((pos, i) => (
        <div
          key={i}
          style={{
            position: "absolute", width: 22, height: 22,
            borderTop: i < 2 ? "2px solid rgba(255,255,255,0.5)" : "none",
            borderBottom: i >= 2 ? "2px solid rgba(255,255,255,0.5)" : "none",
            borderLeft: i % 2 === 0 ? "2px solid rgba(255,255,255,0.5)" : "none",
            borderRight: i % 2 === 1 ? "2px solid rgba(255,255,255,0.5)" : "none",
            ...pos,
          }}
        />
      ))}

      {/* Scan line */}
      {phase === "recording" && (
        <div
          style={{
            position: "absolute", left: 0, right: 0, height: 60,
            background: "linear-gradient(180deg, transparent, oklch(0.55 0.14 245 / 0.4), transparent)",
            animation: "scanline 2s linear infinite",
            mixBlendMode: "screen",
          }}
        />
      )}

      {/* Status row */}
      <div
        style={{
          position: "absolute", top: 14, left: 14, right: 14,
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}
      >
        {phase === "recording" ? (
          <div
            style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "5px 10px", borderRadius: 999,
              background: "rgba(0,0,0,0.5)", backdropFilter: "blur(8px)",
            }}
          >
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "oklch(0.6 0.2 25)", animation: "blink 1s infinite" }} />
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "#fff", letterSpacing: "0.05em" }}>
              REC · {(recordSecs - elapsed).toFixed(1)}s
            </span>
          </div>
        ) : <div />}
        <div
          style={{
            padding: "5px 10px", borderRadius: 999,
            background: "rgba(0,0,0,0.5)", backdropFilter: "blur(8px)",
            fontFamily: "var(--font-mono)", fontSize: 10.5,
            color: "rgba(255,255,255,0.85)", letterSpacing: "0.05em",
          }}
        >
          1080p · 30fps
        </div>
      </div>

      {/* Guidance text */}
      <div
        style={{
          position: "absolute", bottom: 16, left: 16, right: 16,
          textAlign: "center", color: "#fff", fontSize: 12,
          letterSpacing: "-0.005em", textShadow: "0 1px 6px rgba(0,0,0,0.6)",
        }}
      >
        {guidance}
      </div>

      {/* Countdown overlay */}
      {phase === "countdown" && count > 0 && (
        <div
          style={{
            position: "absolute", inset: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
            background: "rgba(0,0,0,0.35)",
          }}
        >
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 88, fontWeight: 300, color: "#fff", textShadow: "0 4px 24px rgba(0,0,0,0.5)" }}>
            {count}
          </div>
        </div>
      )}

      {/* Progress bar */}
      {phase === "recording" && (
        <div
          style={{
            position: "absolute", bottom: 50, left: "50%", transform: "translateX(-50%)",
            width: "calc(100% - 40px)", height: 3, borderRadius: 2, overflow: "hidden",
            background: "rgba(255,255,255,0.15)",
          }}
        >
          <div
            style={{
              width: `${(elapsed / recordSecs) * 100}%`,
              height: "100%", background: "oklch(0.55 0.14 245)",
              transition: "width 0.05s linear",
            }}
          />
        </div>
      )}
    </div>
  );
}

// ── Step: Intro ───────────────────────────────────────────────
function VStepIntro({ onNext, originSite }: { onNext: () => void; originSite: string }) {
  const { t } = useI18n();
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="tag dot" style={{ alignSelf: "flex-start", color: "var(--accent-ink)" }}>
        {t.vRequestFrom} {originSite}
      </div>
      <div>
        <h2 style={{ margin: 0, fontSize: 22, fontWeight: 600, letterSpacing: "-0.02em", lineHeight: 1.2 }}>
          {t.vTitle}
        </h2>
        <p style={{ margin: "10px 0 0", fontSize: 14, color: "var(--muted)", lineHeight: 1.55 }}>
          {t.vDesc}
        </p>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10, padding: 14, background: "var(--bg-2)", borderRadius: 10 }}>
        {[
          ["1", t.vStep1Title, t.vStep1Sub],
          ["2", t.vStep2Title, t.vStep2Sub],
          ["3", t.vStep3Title, t.vStep3Sub],
        ].map(([n, title, sub]) => (
          <div key={n} style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div
              style={{
                width: 22, height: 22, borderRadius: "50%",
                background: "var(--surface)", border: "1px solid var(--line-2)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--ink)", flexShrink: 0,
              }}
            >
              {n}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 500 }}>{title}</div>
              <div style={{ fontSize: 11.5, color: "var(--muted)" }}>{sub}</div>
            </div>
          </div>
        ))}
      </div>
      <button className="btn btn-accent" onClick={onNext} style={{ width: "100%" }}>
        {t.vStartFlow} <Icon.arrowRight size={14} />
      </button>
      <div style={{ fontSize: 10.5, color: "var(--muted-2)", textAlign: "center", fontFamily: "var(--font-sans)" }}>
        {t.vProtected}{Math.floor(Math.random() * 1e6).toString().padStart(6, "0")}
      </div>
    </div>
  );
}

// ── Step: Permission ──────────────────────────────────────────
function VStepPermission({ onNext }: { onNext: () => void }) {
  const { t } = useI18n();
  const [granted, setGranted] = useState(false);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div style={{ alignSelf: "center", marginTop: 8 }}>
        <div
          style={{
            width: 72, height: 72, borderRadius: "50%",
            background: granted ? "var(--ok-soft)" : "var(--accent-soft)",
            color: granted ? "var(--ok)" : "var(--accent-ink)",
            display: "flex", alignItems: "center", justifyContent: "center",
            animation: granted ? "none" : "pulse-ring 2s infinite",
          }}
        >
          {granted ? <Icon.check size={28} /> : <Icon.camera size={28} />}
        </div>
      </div>
      <div style={{ textAlign: "center" }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: "-0.02em" }}>
          {granted ? t.vCamGranted : t.vCamTitle}
        </h2>
        <p style={{ margin: "8px 0 0", fontSize: 13.5, color: "var(--muted)", lineHeight: 1.5 }}>
          {granted ? t.vCamReadyDesc : t.vCamDesc}
        </p>
      </div>
      {!granted ? (
        <button className="btn btn-accent" onClick={() => setGranted(true)} style={{ width: "100%" }}>
          {t.vAllow}
        </button>
      ) : (
        <button className="btn btn-accent" onClick={onNext} style={{ width: "100%" }}>
          {t.vContinue} <Icon.arrowRight size={14} />
        </button>
      )}
    </div>
  );
}

// ── Step: Record ──────────────────────────────────────────────
function VStepRecord({ onNext }: { onNext: () => void }) {
  const { t } = useI18n();
  const [phase, setPhase] = useState<RecordPhase>("idle");
  const handleComplete = () => setTimeout(() => onNext(), 400);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <RecordingViewport phase={phase} onComplete={handleComplete} />
      <div style={{ display: "flex", gap: 8 }}>
        {phase === "idle" && (
          <button
            className="btn btn-accent"
            style={{ flex: 1 }}
            onClick={() => {
              setPhase("countdown");
              setTimeout(() => setPhase("recording"), 2100);
            }}
          >
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#fff" }} />
            {t.vStartRec}
          </button>
        )}
        {(phase === "countdown" || phase === "recording") && (
          <button className="btn btn-ghost" style={{ flex: 1 }} disabled>
            {phase === "countdown" ? t.vPreparing : t.vIsRecording}
          </button>
        )}
      </div>
      <div style={{ fontSize: 11, color: "var(--muted-2)", fontFamily: "var(--font-sans)", textAlign: "center", letterSpacing: "0.04em" }}>
        {t.vCancelNote}
      </div>
    </div>
  );
}

// ── Step: Email ───────────────────────────────────────────────
function VStepEmail({ email, setEmail, onNext }: { email: string; setEmail: (v: string) => void; onNext: () => void }) {
  const { t } = useI18n();
  const valid = /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: "-0.02em" }}>{t.vEmailTitle}</h2>
        <p style={{ margin: "8px 0 0", fontSize: 13.5, color: "var(--muted)", lineHeight: 1.5 }}>{t.vEmailDesc}</p>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <label style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)", letterSpacing: "0.05em", textTransform: "uppercase" }}>
          {t.vEmailLabel}
        </label>
        <input className="txt" type="email" placeholder="name@example.com" value={email} onChange={(e) => setEmail(e.target.value)} autoFocus />
      </div>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: 12, background: "var(--bg-2)", borderRadius: 8, fontSize: 11.5, color: "var(--muted)", lineHeight: 1.5 }}>
        <Icon.shield size={14} />
        <span>
          {t.vHashNote}{" "}
          <span className="mono" style={{ color: "var(--ink-2)" }}>0x4f8a…b2c1</span>
        </span>
      </div>
      <button className="btn btn-accent" onClick={onNext} disabled={!valid} style={{ width: "100%" }}>
        {t.vSendCode} <Icon.arrowRight size={14} />
      </button>
    </div>
  );
}

// ── Step: Code ────────────────────────────────────────────────
function VStepCode({ email, code, setCode, onNext }: { email: string; code: string; setCode: (v: string) => void; onNext: () => void }) {
  const { t } = useI18n();
  const [active, setActive] = useState(0);
  const refs = useRef<(HTMLInputElement | null)[]>([]);

  const setDigit = (i: number, v: string) => {
    const d = v.replace(/[^0-9]/g, "").slice(-1);
    const arr = code.padEnd(6, " ").split("");
    arr[i] = d || " ";
    setCode(arr.join("").trimEnd().slice(0, 6));
    if (d && i < 5) { setActive(i + 1); refs.current[i + 1]?.focus(); }
  };
  const onKey = (i: number, e: React.KeyboardEvent) => {
    if (e.key === "Backspace" && !code[i] && i > 0) { setActive(i - 1); refs.current[i - 1]?.focus(); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: "-0.02em" }}>{t.vCodeTitle}</h2>
        <p style={{ margin: "8px 0 0", fontSize: 13.5, color: "var(--muted)", lineHeight: 1.5 }}>
          {t.vCodeSentTo}{" "}
          <span style={{ color: "var(--ink-2)", fontWeight: 500 }}>{email || "name@example.com"}</span>.{" "}
          {t.vCodeHint}{" "}
          <span className="mono" style={{ color: "var(--accent-ink)" }}>284 619</span>
        </p>
      </div>
      <div style={{ display: "flex", gap: 8, justifyContent: "space-between" }}>
        {Array.from({ length: 6 }).map((_, i) => (
          <input
            key={i}
            ref={(el) => { refs.current[i] = el; }}
            value={code[i] || ""}
            onChange={(e) => setDigit(i, e.target.value)}
            onKeyDown={(e) => onKey(i, e)}
            onFocus={() => setActive(i)}
            maxLength={1}
            inputMode="numeric"
            style={{
              flex: 1, height: 52, textAlign: "center",
              fontFamily: "var(--font-mono)", fontSize: 22, fontWeight: 500,
              border: `1.5px solid ${active === i ? "var(--accent)" : "var(--line-2)"}`,
              borderRadius: 8, background: "var(--surface)", color: "var(--ink)", outline: "none",
              boxShadow: active === i ? "0 0 0 3px oklch(0.55 0.14 245 / 0.15)" : "none",
              transition: "all 0.12s",
            }}
          />
        ))}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 12 }}>
        <span style={{ color: "var(--muted)" }}>{t.vNoCode}</span>
        <button className="btn btn-ghost" style={{ padding: "6px 10px", fontSize: 12 }}>
          <Icon.refresh size={12} /> {t.vResend}
        </button>
      </div>
      <button className="btn btn-accent" onClick={onNext} disabled={code.trim().length !== 6} style={{ width: "100%" }}>
        {t.vConfirm} <Icon.arrowRight size={14} />
      </button>
    </div>
  );
}

// ── Step: Processing ──────────────────────────────────────────
function VStepProcessing({ onNext }: { onNext: () => void }) {
  const { t } = useI18n();
  const durations = [800, 1200, 1100, 900, 600];
  const models = ["pipeline", "spatial", "temporal", "liveness", "fusion"];
  const [stage, setStage] = useState(0);

  useEffect(() => {
    if (stage < durations.length) {
      const id = setTimeout(() => setStage(stage + 1), durations[stage]);
      return () => clearTimeout(id);
    } else {
      const id = setTimeout(() => onNext(), 400);
      return () => clearTimeout(id);
    }
  }, [stage, onNext]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div style={{ alignSelf: "center", marginTop: 8 }}>
        <div
          style={{
            width: 56, height: 56, borderRadius: "50%",
            background: "var(--accent-soft)", color: "var(--accent-ink)",
            display: "flex", alignItems: "center", justifyContent: "center", position: "relative",
          }}
        >
          <div className="spinner" style={{ width: 56, height: 56, position: "absolute", inset: 0, borderWidth: 2 }} />
          <Icon.cpu size={22} />
        </div>
      </div>
      <div style={{ textAlign: "center" }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: "-0.02em" }}>{t.vAnalyzingTitle}</h2>
        <p style={{ margin: "8px 0 0", fontSize: 13.5, color: "var(--muted)", lineHeight: 1.5 }}>{t.vAnalyzingDesc}</p>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8, padding: 14, background: "var(--bg-2)", borderRadius: 10 }}>
        {t.vStages.map((label, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12.5 }}>
            <div
              style={{
                width: 14, height: 14, borderRadius: "50%",
                background: i < stage ? "var(--ok)" : i === stage ? "transparent" : "var(--line)",
                border: i === stage ? "2px solid var(--accent)" : "none",
                display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
              }}
            >
              {i < stage && <Icon.check size={9} style={{ color: "#fff" }} />}
              {i === stage && <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent)", animation: "blink 1s infinite" }} />}
            </div>
            <span className="mono" style={{ color: i <= stage ? "var(--ink-2)" : "var(--muted-2)", flex: 1, fontSize: 11.5 }}>
              {label}
            </span>
            <span className="mono" style={{ color: "var(--muted-2)", fontSize: 10.5 }}>{models[i]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Step: Result ──────────────────────────────────────────────
function VStepResult({ result, onRetry, onReturn, originSite }: { result: "pass" | "retry"; onRetry: () => void; onReturn: () => void; originSite: string }) {
  const { t } = useI18n();

  if (result === "pass") {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        <div style={{ alignSelf: "center", marginTop: 8 }}>
          <div style={{ width: 72, height: 72, borderRadius: "50%", background: "var(--ok-soft)", color: "var(--ok)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Icon.check size={32} />
          </div>
        </div>
        <div style={{ textAlign: "center" }}>
          <h2 style={{ margin: 0, fontSize: 22, fontWeight: 600, letterSpacing: "-0.02em" }}>{t.vPassTitle}</h2>
          <p style={{ margin: "8px 0 0", fontSize: 13.5, color: "var(--muted)", lineHeight: 1.5 }}>{t.vPassDesc}</p>
        </div>
        <div className="card" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 10 }}>
          {([
            [t.vPassConfidence, "99.4%", "var(--ok)"],
            [t.vPassLiveness, "0.97", "var(--ok)"],
            [t.vPassTime, "4.6s", "var(--muted)"],
            [t.vPassModels, "3 / 3", "var(--muted)"],
          ] as [string, string, string][]).map(([k, v, c]) => (
            <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}>
              <span style={{ color: "var(--muted)" }}>{k}</span>
              <span className="mono" style={{ color: c, fontWeight: 500 }}>{v}</span>
            </div>
          ))}
        </div>
        <button className="btn btn-accent" onClick={onReturn} style={{ width: "100%" }}>
          {t.vReturnTo} {originSite} <Icon.arrowRight size={14} />
        </button>
        <div style={{ textAlign: "center", fontSize: 11, color: "var(--muted-2)", fontFamily: "var(--font-sans)" }}>
          {t.vAutoReturn}
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div style={{ alignSelf: "center", marginTop: 8 }}>
        <div style={{ width: 72, height: 72, borderRadius: "50%", background: "var(--danger-soft)", color: "var(--danger)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Icon.warn size={28} />
        </div>
      </div>
      <div style={{ textAlign: "center" }}>
        <h2 style={{ margin: 0, fontSize: 22, fontWeight: 600, letterSpacing: "-0.02em" }}>{t.vRetryTitle}</h2>
        <p style={{ margin: "8px 0 0", fontSize: 13.5, color: "var(--muted)", lineHeight: 1.5 }}>{t.vRetryDesc}</p>
      </div>
      <div className="card" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 10 }}>
        {([
          [t.vAnom, t.vHigh, "var(--danger)"],
          [t.vLiveness, "0.21", "var(--danger)"],
          [t.vLipSync, t.vModerate, "var(--warn)"],
        ] as [string, string, string][]).map(([k, v, c]) => (
          <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}>
            <span style={{ color: "var(--muted)" }}>{k}</span>
            <span className="mono" style={{ color: c, fontWeight: 500 }}>{v}</span>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <button className="btn btn-accent" onClick={onRetry} style={{ width: "100%" }}>
          <Icon.refresh size={14} /> {t.vRetryBtn}
        </button>
        <button className="btn btn-ghost" onClick={onReturn} style={{ width: "100%" }}>
          {t.vCancelReturn}
        </button>
      </div>
    </div>
  );
}

// ── Flow shell ────────────────────────────────────────────────
type Step = "intro" | "permission" | "record" | "email" | "code" | "processing" | "result";
const STEP_ORDER: Step[] = ["intro", "permission", "record", "email", "code", "processing", "result"];

export function VerificationFlow({
  start = "intro",
  forceResult = null,
  originSite = "examplebank.kz",
  compact = false,
}: {
  start?: Step;
  forceResult?: "pass" | "retry" | null;
  originSite?: string;
  compact?: boolean;
}) {
  const [step, setStep] = useState<Step>(start);
  const [email, setEmail] = useState("user@example.com");
  const [code, setCode] = useState("284619");
  const [result] = useState<"pass" | "retry">(forceResult ?? (Math.random() > 0.5 ? "pass" : "retry"));

  const idx = STEP_ORDER.indexOf(step);
  const next = () => {
    const cur = STEP_ORDER.indexOf(step);
    if (cur < STEP_ORDER.length - 1) setStep(STEP_ORDER[cur + 1]);
  };
  const restart = () => setStep("record");

  return (
    <div style={{ display: "flex", flexDirection: "column", padding: compact ? "20px 20px 16px" : "24px 24px 20px", gap: 16, height: "100%" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <Logo size={13} />
        <div style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
          {idx + 1}/{STEP_ORDER.length}
        </div>
      </div>

      {/* Progress */}
      <div style={{ height: 3, borderRadius: 2, background: "var(--bg-2)", overflow: "hidden" }}>
        <div
          style={{
            width: `${((idx + 1) / STEP_ORDER.length) * 100}%`,
            height: "100%", background: "var(--accent)",
            transition: "width 0.3s cubic-bezier(0.2, 0.7, 0.3, 1)",
          }}
        />
      </div>

      {/* Body */}
      <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
        {step === "intro" && <VStepIntro onNext={next} originSite={originSite} />}
        {step === "permission" && <VStepPermission onNext={next} />}
        {step === "record" && <VStepRecord onNext={next} />}
        {step === "email" && <VStepEmail email={email} setEmail={setEmail} onNext={next} />}
        {step === "code" && <VStepCode email={email} code={code} setCode={setCode} onNext={next} />}
        {step === "processing" && <VStepProcessing onNext={next} />}
        {step === "result" && <VStepResult result={result} onRetry={restart} onReturn={() => setStep("intro")} originSite={originSite} />}
      </div>
    </div>
  );
}
