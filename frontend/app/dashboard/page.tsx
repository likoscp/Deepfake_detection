"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/auth-provider";
import { Dashboard } from "@/components/dashboard";

export default function DashboardPage() {
  const { session, loaded } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (loaded && !session) router.replace("/login");
  }, [loaded, session, router]);

  if (!loaded) {
    return (
      <div
        style={{
          height: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "var(--bg)",
        }}
      >
        <span className="spinner" style={{ width: 28, height: 28 }} />
      </div>
    );
  }

  if (!session) return null;

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Dashboard initialPane="overview" />
    </div>
  );
}
