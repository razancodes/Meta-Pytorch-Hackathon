import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MEMEX — Financial Crime Intelligence Terminal",
  description: "Real-time AML investigation and threat intelligence platform. Autonomous agent monitoring with OS-level mechanics visualization.",
  keywords: ["AML", "financial crime", "intelligence", "investigation", "money laundering", "memex"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={jetbrainsMono.variable}>
      <body style={{ fontFamily: "var(--font-jetbrains), 'JetBrains Mono', monospace" }}>
        <div className="nx-scanline-overlay" />
        {children}
      </body>
    </html>
  );
}
