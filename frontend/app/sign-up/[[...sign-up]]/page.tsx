/**
 * Hosted sign-up route. See app/sign-in/[[...sign-in]]/page.tsx for
 * rationale — same pattern, opposite flow.
 */

import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
  return (
    <main className="mx-auto flex w-full max-w-md flex-col items-center px-4 py-12">
      <SignUp />
    </main>
  );
}
