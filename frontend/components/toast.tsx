"use client";

import React, { createContext, useContext, useState, useCallback, useRef } from "react";
import { Icon } from "./icons";

type ToastType = "success" | "error" | "info";

interface Toast {
  id: number;
  msg: string;
  type: ToastType;
}

interface ToastCtx {
  show: (msg: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastCtx>({ show: () => {} });

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const idRef = useRef(0);

  const show = useCallback((msg: string, type: ToastType = "success") => {
    const id = idRef.current++;
    setToasts((prev) => [...prev, { id, msg, type }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 3200);
  }, []);

  return (
    <ToastContext.Provider value={{ show }}>
      {children}
      {toasts.length > 0 && (
        <div
          style={{
            position: "fixed",
            bottom: 24,
            right: 24,
            zIndex: 9999,
            display: "flex",
            flexDirection: "column",
            gap: 8,
            pointerEvents: "none",
          }}
        >
          {toasts.map((t) => (
            <div
              key={t.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "12px 16px",
                borderRadius: 10,
                background: "var(--surface)",
                border: "1px solid var(--line)",
                boxShadow: "0 8px 24px rgba(0,0,0,0.15)",
                fontSize: 13,
                color: "var(--ink)",
                animation: "slide-in-up 0.18s ease-out",
                pointerEvents: "auto",
                minWidth: 220,
              }}
            >
              {t.type === "success" && (
                <Icon.check size={15} style={{ color: "var(--ok)", flexShrink: 0 }} />
              )}
              {t.type === "error" && (
                <Icon.warn size={15} style={{ color: "var(--danger)", flexShrink: 0 }} />
              )}
              {t.type === "info" && (
                <Icon.spark size={15} style={{ color: "var(--accent-ink)", flexShrink: 0 }} />
              )}
              <span>{t.msg}</span>
            </div>
          ))}
        </div>
      )}
    </ToastContext.Provider>
  );
}

export function useToast() {
  return useContext(ToastContext);
}
