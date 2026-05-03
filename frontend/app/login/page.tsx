"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Logo } from "@/components/primitives";
import { useI18n, LangSwitcher } from "@/components/i18n-provider";
import { ThemeToggle } from "@/components/theme-provider";
import { useAuth } from "@/components/auth-provider";

// ── 6-box OTP input ───────────────────────────────────────────
const MOCK_CODE = "123456";

function OtpInput({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const refs = useRef<(HTMLInputElement | null)[]>([]);

  const handleChange = (i: number, raw: string) => {
    const digit = raw.replace(/\D/g, "").slice(-1);
    const arr = value.padEnd(6, " ").split("").map((c) => (c === " " ? "" : c));
    arr[i] = digit;
    onChange(arr.join(""));
    if (digit && i < 5) refs.current[i + 1]?.focus();
  };

  const handleKeyDown = (i: number, e: React.KeyboardEvent) => {
    if (e.key === "Backspace") {
      e.preventDefault();
      const arr = value.padEnd(6, " ").split("").map((c) => (c === " " ? "" : c));
      if (arr[i]) {
        arr[i] = "";
        onChange(arr.join(""));
      } else if (i > 0) {
        arr[i - 1] = "";
        onChange(arr.join(""));
        refs.current[i - 1]?.focus();
      }
    } else if (e.key === "ArrowLeft" && i > 0) refs.current[i - 1]?.focus();
    else if (e.key === "ArrowRight" && i < 5) refs.current[i + 1]?.focus();
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const digits = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
    onChange(digits.padEnd(6, "").slice(0, 6).replace(/ /g, ""));
    refs.current[Math.min(digits.length, 5)]?.focus();
  };

  return (
    <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
      {Array.from({ length: 6 }, (_, i) => {
        const filled = !!(value[i] && value[i] !== " ");
        return (
          <input
            key={i}
            ref={(el) => { refs.current[i] = el; }}
            type="text"
            inputMode="numeric"
            maxLength={2}
            value={value[i] && value[i] !== " " ? value[i] : ""}
            onChange={(e) => handleChange(i, e.target.value)}
            onKeyDown={(e) => handleKeyDown(i, e)}
            onPaste={handlePaste}
            onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
            onBlur={(e) => (e.currentTarget.style.borderColor = filled ? "var(--accent-ink)" : "var(--line-2)")}
            style={{
              width: 46,
              height: 56,
              textAlign: "center",
              fontSize: 22,
              fontWeight: 700,
              fontFamily: "var(--font-mono)",
              color: "var(--ink)",
              background: "var(--surface)",
              border: `1.5px solid ${filled ? "var(--accent-ink)" : "var(--line-2)"}`,
              borderRadius: 10,
              outline: "none",
              transition: "border-color 0.12s, background 0.12s",
              cursor: "text",
            }}
          />
        );
      })}
    </div>
  );
}

// ── Login page ────────────────────────────────────────────────
export default function LoginPage() {
  const { t } = useI18n();
  const { verifyCredentials, completeLogin, session, loaded } = useAuth();
  const router = useRouter();

  const [step, setStep] = useState<1 | 2>(1);
  const [loginId, setLoginId] = useState("");
  const [password, setPassword] = useState("");
  const [maskedEmail, setMaskedEmail] = useState("");
  const [pendingLogin, setPendingLogin] = useState("");
  const [otp, setOtp] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [resent, setResent] = useState(false);

  useEffect(() => {
    if (loaded && session) router.replace("/dashboard");
  }, [loaded, session, router]);

  // auto-submit when all 6 digits entered
  useEffect(() => {
    if (otp.replace(/\s/g, "").length === 6 && step === 2) {
      handleVerify(otp);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [otp]);

  const handleStep1 = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (!loginId.trim() || !password.trim()) { setError(t.authErrRequired); return; }
    setLoading(true);
    await new Promise((r) => setTimeout(r, 400));
    const result = verifyCredentials(loginId.trim(), password);
    setLoading(false);
    if (!result.ok) { setError(t.authErrInvalid); return; }
    setPendingLogin(loginId.trim());
    setMaskedEmail(result.maskedEmail ?? "");
    setStep(2);
    setOtp("");
  };

  const handleVerify = async (code = otp) => {
    setError("");
    const clean = code.replace(/\s/g, "");
    if (clean.length < 6) { setError(t.auth2faWrong); return; }
    setLoading(true);
    await new Promise((r) => setTimeout(r, 500));
    // TODO: replace with real API call: POST /api/auth/verify-otp
    if (clean !== MOCK_CODE) {
      setLoading(false);
      setError(t.auth2faWrong);
      setOtp("");
      return;
    }
    completeLogin(pendingLogin);
    router.push("/dashboard");
  };

  const handleResend = async () => {
    setResent(false);
    await new Promise((r) => setTimeout(r, 600));
    // TODO: real API call: POST /api/auth/resend-otp
    setResent(true);
    setTimeout(() => setResent(false), 3000);
  };

  if (!loaded || session) return null;

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", display: "flex", flexDirection: "column" }}>
      {/* Topbar */}
      <div
        style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "16px 32px", borderBottom: "1px solid var(--line)", background: "var(--surface)",
        }}
      >
        <Link href="/" style={{ textDecoration: "none", color: "inherit" }}>
          <Logo size={14} />
        </Link>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <LangSwitcher />
          <ThemeToggle />
        </div>
      </div>

      {/* Card */}
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "40px 24px" }}>
        <div
          style={{
            width: "100%", maxWidth: 400,
            background: "var(--surface)", border: "1px solid var(--line)",
            borderRadius: 16, padding: "40px 36px",
          }}
        >
          {/* Logo + title */}
          <div style={{ textAlign: "center", marginBottom: 32 }}>
            <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
              <Logo size={15} />
            </div>
            <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em", color: "var(--ink)", marginBottom: 8 }}>
              {step === 1 ? t.authLoginTitle : t.auth2faTitle}
            </div>
            <div style={{ fontSize: 13.5, color: "var(--muted)", lineHeight: 1.5 }}>
              {step === 1
                ? t.authLoginSub
                : `${t.auth2faSub} ${maskedEmail}`}
            </div>
          </div>

          {/* Step 1: credentials */}
          {step === 1 && (
            <form onSubmit={handleStep1} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <label style={{ fontSize: 12, fontWeight: 500, color: "var(--ink-2)", letterSpacing: "0.02em" }}>
                  {t.authLoginField}
                </label>
                <input
                  className="txt" type="text" autoComplete="username" autoFocus
                  value={loginId} onChange={(e) => setLoginId(e.target.value)} placeholder="admin"
                />
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <label style={{ fontSize: 12, fontWeight: 500, color: "var(--ink-2)", letterSpacing: "0.02em" }}>
                  {t.authPassField}
                </label>
                <input
                  className="txt" type="password" autoComplete="current-password"
                  value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••"
                />
              </div>
              {error && <ErrorBox msg={error} />}
              <button
                type="submit" className="btn btn-primary" disabled={loading}
                style={{ width: "100%", marginTop: 4, padding: "12px 0", fontSize: 14 }}
              >
                {loading ? <span className="spinner" style={{ width: 16, height: 16 }} /> : t.authLoginBtn}
              </button>
            </form>
          )}

          {/* Step 2: OTP */}
          {step === 2 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <OtpInput value={otp} onChange={setOtp} />

              {error && <ErrorBox msg={error} />}

              <button
                className="btn btn-primary" disabled={loading || otp.replace(/\s/g, "").length < 6}
                onClick={() => handleVerify()}
                style={{ width: "100%", padding: "12px 0", fontSize: 14 }}
              >
                {loading ? <span className="spinner" style={{ width: 16, height: 16 }} /> : t.auth2faBtn}
              </button>

              {/* Resend */}
              <div style={{ textAlign: "center" }}>
                {resent ? (
                  <span style={{ fontSize: 12.5, color: "var(--ok)" }}>✓ Code resent</span>
                ) : (
                  <button
                    onClick={handleResend}
                    style={{ background: "none", border: 0, cursor: "pointer", fontSize: 12.5, color: "var(--accent-ink)", fontFamily: "var(--font-sans)" }}
                  >
                    {t.auth2faResend}
                  </button>
                )}
              </div>

              {/* Back */}
              <button
                onClick={() => { setStep(1); setError(""); setOtp(""); }}
                style={{ background: "none", border: 0, cursor: "pointer", fontSize: 12.5, color: "var(--muted)", fontFamily: "var(--font-sans)", textAlign: "center" }}
              >
                {t.auth2faBack}
              </button>
            </div>
          )}

          {/* Dev hint */}
          <div
            style={{
              marginTop: 20, padding: "9px 12px",
              background: "var(--accent-soft)", borderRadius: 8,
              fontSize: 12, color: "var(--accent-ink)", textAlign: "center",
            }}
          >
            {step === 1 ? t.authHint : t.auth2faDevHint}
          </div>

          {/* Register link */}
          {step === 1 && (
            <div style={{ marginTop: 20, textAlign: "center", fontSize: 13, color: "var(--muted)" }}>
              {t.authNoAcc}{" "}
              <Link href="/register" style={{ color: "var(--accent-ink)", fontWeight: 500, textDecoration: "none" }}>
                {t.authRegisterLink}
              </Link>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ErrorBox({ msg }: { msg: string }) {
  return (
    <div
      style={{
        fontSize: 12.5, color: "var(--danger)",
        background: "var(--danger-soft)",
        border: "1px solid oklch(0.58 0.18 25 / 0.2)",
        borderRadius: 8, padding: "9px 12px",
      }}
    >
      {msg}
    </div>
  );
}
