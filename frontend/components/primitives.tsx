"use client";

import React from "react";
import { ShieldIcon } from "./icons";

// ── Logo ──────────────────────────────────────────────────────
export function Logo({ dark = false, size = 16 }: { dark?: boolean; size?: number }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        fontFamily: "var(--font-mono)",
        fontSize: size,
        fontWeight: 600,
        letterSpacing: "-0.01em",
        color: dark ? "#fff" : "var(--ink)",
      }}
    >
      <div
        style={{
          width: size + 4,
          height: size + 4,
          borderRadius: 4,
          background: "linear-gradient(135deg, oklch(0.55 0.14 245), oklch(0.42 0.15 245))",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#fff",
        }}
      >
        <ShieldIcon size={size - 2} strokeWidth="2.4" />
      </div>
      <span>
        verus<span style={{ color: "oklch(0.55 0.14 245)" }}>/</span>id
      </span>
    </div>
  );
}

// ── Sparkline ─────────────────────────────────────────────────
export function Sparkline({
  data,
  color = "var(--accent)",
  width = 120,
  height = 32,
  fill = true,
}: {
  data: number[];
  color?: string;
  width?: number;
  height?: number;
  fill?: boolean;
}) {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const stepX = width / (data.length - 1);
  const points = data.map((d, i) => [i * stepX, height - ((d - min) / range) * (height - 4) - 2]);
  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const fillPath = `${path} L${width} ${height} L0 ${height} Z`;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
      {fill && <path d={fillPath} fill={color} fillOpacity="0.12" />}
      <path d={path} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── BarChart ──────────────────────────────────────────────────
export function BarChart({
  data,
  height = 120,
}: {
  data: { label: string; value: number; highlight?: boolean }[];
  height?: number;
}) {
  const max = Math.max(...data.map((d) => d.value));
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 6, height, width: "100%" }}>
      {data.map((d, i) => (
        <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 6, height: "100%" }}>
          <div
            style={{
              width: "100%",
              height: `${(d.value / max) * 100}%`,
              background: d.highlight ? "var(--accent)" : "oklch(0.55 0.14 245 / 0.18)",
              borderRadius: "4px 4px 0 0",
              minHeight: 2,
            }}
          />
          <div style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>{d.label}</div>
        </div>
      ))}
    </div>
  );
}

// ── Donut ─────────────────────────────────────────────────────
export function Donut({
  segments,
  size = 120,
  thickness = 18,
}: {
  segments: { value: number; color: string }[];
  size?: number;
  thickness?: number;
}) {
  const total = segments.reduce((s, x) => s + x.value, 0);
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  let offset = 0;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--bg-2)" strokeWidth={thickness} />
      {segments.map((s, i) => {
        const len = (s.value / total) * c;
        const dash = `${len} ${c - len}`;
        const dashOffset = -offset;
        offset += len;
        return (
          <circle
            key={i}
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={s.color}
            strokeWidth={thickness}
            strokeDasharray={dash}
            strokeDashoffset={dashOffset}
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
          />
        );
      })}
    </svg>
  );
}

// ── Stat ──────────────────────────────────────────────────────
export function Stat({
  label,
  value,
  sub,
  trend,
}: {
  label: string;
  value: string;
  sub?: string;
  trend?: "up" | "down" | null;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div
        style={{
          fontSize: 11,
          color: "var(--muted)",
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          fontFamily: "var(--font-sans)",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 28,
          fontWeight: 500,
          letterSpacing: "-0.02em",
          color: "var(--ink)",
          fontFamily: "var(--font-mono)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
      {sub && (
        <div
          style={{
            fontSize: 11.5,
            color: trend === "up" ? "var(--ok)" : trend === "down" ? "var(--danger)" : "var(--muted)",
            fontFamily: "var(--font-sans)",
          }}
        >
          {sub}
        </div>
      )}
    </div>
  );
}

// ── Avatar ────────────────────────────────────────────────────
export function Avatar({ name = "AC", size = 32, hue = 245 }: { name?: string; size?: number; hue?: number }) {
  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        background: `oklch(0.92 0.04 ${hue})`,
        color: `oklch(0.42 0.12 ${hue})`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: size * 0.36,
        fontWeight: 600,
        letterSpacing: "0.02em",
        fontFamily: "var(--font-mono)",
        flexShrink: 0,
      }}
    >
      {name}
    </div>
  );
}

// ── Toggle ────────────────────────────────────────────────────
export function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!value)}
      style={{
        appearance: "none",
        border: 0,
        cursor: "pointer",
        width: 40,
        height: 24,
        borderRadius: 999,
        background: value ? "var(--accent)" : "var(--line-2)",
        position: "relative",
        transition: "background 0.15s",
        flexShrink: 0,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 3,
          left: value ? 19 : 3,
          width: 18,
          height: 18,
          borderRadius: "50%",
          background: "#fff",
          transition: "left 0.15s",
          boxShadow: "0 1px 3px rgba(0,0,0,0.15)",
        }}
      />
    </button>
  );
}
