import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MicroGhost Thermal Analyzer",
  description: "RGB and thermal intrusion detection powered by MicroGhost."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
