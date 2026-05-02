"use client";

import { VerificationFlow } from "@/components/verification-flow";

export default function VerifyPage() {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0e1726",
        backgroundImage: "radial-gradient(ellipse at top, oklch(0.55 0.14 245 / 0.25), transparent 60%)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
      }}
    >
      <div
        className="card"
        style={{
          width: 420,
          maxHeight: "calc(100vh - 48px)",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          boxShadow: "0 24px 80px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05)",
        }}
      >
        <VerificationFlow originSite="examplebank.kz" />
      </div>
    </div>
  );
}
