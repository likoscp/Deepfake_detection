"use client";

import React, { createContext, useContext, useState } from "react";
import { translations, type Lang, type T } from "@/lib/translations";

interface I18nCtx {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: T;
}

const I18nContext = createContext<I18nCtx>({
  lang: "en",
  setLang: () => {},
  t: translations.en,
});

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLang] = useState<Lang>("en");
  return (
    <I18nContext.Provider value={{ lang, setLang, t: translations[lang] }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  return useContext(I18nContext);
}

// ── Language switcher pill ────────────────────────────────────
export function LangSwitcher() {
  const { lang, setLang } = useI18n();
  const opts: { value: Lang; label: string }[] = [
    { value: "en", label: "EN" },
    { value: "ru", label: "RU" },
    { value: "kz", label: "ҚАЗ" },
  ];
  return (
    <div
      style={{
        display: "flex",
        borderRadius: 8,
        border: "1px solid var(--line-2)",
        overflow: "hidden",
        background: "var(--surface)",
      }}
    >
      {opts.map((o) => (
        <button
          key={o.value}
          onClick={() => setLang(o.value)}
          style={{
            appearance: "none",
            border: 0,
            cursor: "pointer",
            padding: "6px 12px",
            fontSize: 12,
            fontWeight: 500,
            fontFamily: "var(--font-sans)",
            letterSpacing: "0.02em",
            background: lang === o.value ? "var(--ink)" : "transparent",
            color: lang === o.value ? "var(--bg)" : "var(--muted)",
            transition: "background 0.12s, color 0.12s",
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
