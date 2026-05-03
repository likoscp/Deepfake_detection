"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

export interface User {
  login: string;
  password: string;
  companyName: string;
  email: string;
}

export interface Session {
  login: string;
  companyName: string;
  email: string;
}

interface AuthCtx {
  session: Session | null;
  loaded: boolean;
  /** Step 1 of login: verify credentials, returns masked email on success */
  verifyCredentials: (login: string, password: string) => { ok: boolean; maskedEmail?: string; error?: string };
  /** Step 2 of login: finalize after OTP verified on server */
  completeLogin: (login: string) => void;
  /** Step 2 of register: create account after OTP verified */
  register: (companyName: string, login: string, email: string, password: string) => { ok: boolean; error?: string };
  logout: () => void;
}

const SEED: User[] = [
  { login: "admin", password: "admin", companyName: "Admin Corp", email: "admin@verus.id" },
];

function maskEmail(email: string | undefined): string {
  if (!email) return "your email";
  const [local, domain] = email.split("@");
  if (!domain) return email;
  const visible = local.slice(0, 2);
  return `${visible}${"*".repeat(Math.max(local.length - 2, 2))}@${domain}`;
}

function getUsers(): User[] {
  try {
    const raw = localStorage.getItem("verus-users");
    let stored: User[] = raw ? JSON.parse(raw) : [];
    // migrate: ensure every user has an email field
    stored = stored.map((u) => ({ ...u, email: u.email ?? "" }));
    const adminIdx = stored.findIndex((u) => u.login === "admin");
    if (adminIdx === -1) {
      stored.unshift(SEED[0]);
    } else if (!stored[adminIdx].email) {
      stored[adminIdx] = { ...stored[adminIdx], email: SEED[0].email };
    }
    return stored;
  } catch {
    return SEED;
  }
}

function saveUsers(users: User[]) {
  localStorage.setItem("verus-users", JSON.stringify(users));
}

const AuthContext = createContext<AuthCtx>({
  session: null,
  loaded: false,
  verifyCredentials: () => ({ ok: false }),
  completeLogin: () => {},
  register: () => ({ ok: false }),
  logout: () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem("verus-session");
      if (raw) setSession(JSON.parse(raw));
    } catch {}
    const users = getUsers();
    saveUsers(users);
    setLoaded(true);
  }, []);

  const verifyCredentials = (loginId: string, password: string) => {
    const users = getUsers();
    const user = users.find((u) => u.login === loginId && u.password === password);
    if (!user) return { ok: false, error: "invalid" };
    return { ok: true, maskedEmail: maskEmail(user.email) };
  };

  const completeLogin = (loginId: string) => {
    const users = getUsers();
    const user = users.find((u) => u.login === loginId);
    if (!user) return;
    const sess: Session = { login: user.login, companyName: user.companyName, email: user.email };
    localStorage.setItem("verus-session", JSON.stringify(sess));
    setSession(sess);
  };

  const register = (companyName: string, loginId: string, email: string, password: string) => {
    const users = getUsers();
    if (users.find((u) => u.login === loginId)) return { ok: false, error: "exists" };
    const newUser: User = { login: loginId, password, companyName, email };
    saveUsers([...users, newUser]);
    const sess: Session = { login: loginId, companyName, email };
    localStorage.setItem("verus-session", JSON.stringify(sess));
    setSession(sess);
    return { ok: true };
  };

  const logout = () => {
    localStorage.removeItem("verus-session");
    setSession(null);
  };

  return (
    <AuthContext.Provider value={{ session, loaded, verifyCredentials, completeLogin, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
