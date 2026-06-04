/**
 * Hosted sign-in route. The nav uses `mode="modal"` so this page is
 * the fallback when someone hits /sign-in directly — e.g. via a deep
 * link in an email — and also where Clerk redirects on OAuth callbacks.
 *
 * The `[[...sign-in]]` catch-all is Clerk's expected pattern: the
 * library mounts sub-routes for password recovery, MFA challenges,
 * organisation selection, etc., underneath this segment.
 */

import { SignIn } from "@clerk/nextjs";

export default function SignInPage() {
  return (
    <main className="mx-auto flex w-full max-w-md flex-col items-center px-4 py-12">
      <SignIn />
    </main>
  );
}
