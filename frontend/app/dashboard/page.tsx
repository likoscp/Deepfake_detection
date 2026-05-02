"use client";

import { Dashboard } from "@/components/dashboard";

export default function DashboardPage() {
  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Dashboard initialPane="overview" />
    </div>
  );
}
