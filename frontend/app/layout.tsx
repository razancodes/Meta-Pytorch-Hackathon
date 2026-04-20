import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Memex — AML Intelligence Terminal",
  description: "Glass Box visualizer for autonomous RL agent operating system. Anti-Money Laundering investigation platform.",
  keywords: ["AML", "Anti-Money Laundering", "Reinforcement Learning", "1MDB", "Investigation"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-bg text-text min-h-screen antialiased" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
        {children}
      </body>
    </html>
  );
}
