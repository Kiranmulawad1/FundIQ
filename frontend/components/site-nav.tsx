import { Show, SignInButton, SignUpButton, UserButton } from "@clerk/nextjs";
import Link from "next/link";

import { SavedNavLink } from "@/components/saved-nav-link";
import { ThemeToggle } from "@/components/theme-toggle";

export function SiteNav() {
  return (
    <nav className="border-b border-border/60 bg-background/80 backdrop-blur print:hidden">
      <div className="mx-auto flex w-full max-w-4xl items-baseline gap-6 px-4 py-3">
        <Link
          href="/"
          className="text-base font-semibold tracking-tight hover:opacity-80"
        >
          FundIQ
        </Link>
        <Link
          href="/"
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          Search
        </Link>
        <Link
          href="/recommend"
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          Recommend
        </Link>
        <Link
          href="/profile"
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          Profile
        </Link>
        <Link
          href="/grants"
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          Browse
        </Link>
        <Link
          href="/analytics"
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          Analytics
        </Link>
        <SavedNavLink />
        <span className="ml-auto hidden text-xs text-muted-foreground/60 lg:inline">
          AI funding intelligence — EU &amp; German grants
        </span>
        <ThemeToggle />
        <ClerkAuthCluster />
      </div>
    </nav>
  );
}

/**
 * Auth controls — three states, all rendered server-side from Clerk's
 * session cookie so the right element is in the initial HTML and we
 * dodge a hydration flicker.
 *
 *   Signed out  → "Sign in" + "Sign up" buttons (modal forces stay on page)
 *   Signed in   → <UserButton/> avatar (opens Clerk's account menu)
 */
function ClerkAuthCluster() {
  // Clerk 7 collapsed the old <SignedIn/> and <SignedOut/> control
  // components into a single <Show when="..."> primitive. Same runtime
  // gating, fewer exports.
  return (
    <>
      <Show when="signed-out">
        <SignInButton mode="modal">
          <button
            type="button"
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            Sign in
          </button>
        </SignInButton>
        <SignUpButton mode="modal">
          <button
            type="button"
            className="rounded-md border border-foreground/15 bg-foreground/[0.04] px-3 py-1 text-sm hover:bg-foreground/[0.08]"
          >
            Sign up
          </button>
        </SignUpButton>
      </Show>
      <Show when="signed-in">
        <UserButton
          // Tighten the avatar so it doesn't overflow our 36px-tall row.
          appearance={{ elements: { userButtonAvatarBox: "h-7 w-7" } }}
        />
      </Show>
    </>
  );
}
