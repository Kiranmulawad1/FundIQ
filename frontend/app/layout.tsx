import { ClerkProvider } from "@clerk/nextjs";
import { shadcn } from "@clerk/ui/themes";
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { SiteNav } from "@/components/site-nav";
import { ThemeProvider } from "@/components/theme-provider";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FundIQ — AI funding intelligence",
  description:
    "Hybrid retrieval over EU and German startup grants. EXIST, EIC, KfW, regional programmes.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  // ClerkProvider goes inside <body>, not wrapping <html> — per Clerk's
  // Next.js 15+ guide. Putting it around <html> works in dev but breaks
  // a few server-component code paths (`auth()` from the layout, etc.).
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full bg-background font-sans text-foreground">
        <ClerkProvider appearance={{ theme: shadcn }}>
          <ThemeProvider>
            <SiteNav />
            {children}
          </ThemeProvider>
        </ClerkProvider>
      </body>
    </html>
  );
}
