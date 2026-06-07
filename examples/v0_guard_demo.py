from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_runtime import (  # noqa: E402
    LeakageVerifier,
    PrivacyGuardV0,
    PrivacyPolicy,
    ProtectedFact,
    SemanticLeakageVerifierV0,
    TemplateAbstractionRewriter,
    build_counterfactual_text,
)


def build_policy() -> PrivacyPolicy:
    return PrivacyPolicy(
        facts=(
            ProtectedFact(
                fact_id="budget",
                fact="budget ceiling is $2M",
                surface_forms=("$2M", "2M"),
                semantic_hints=("maximum budget", "cannot exceed"),
                allowed_abstractions=("a constrained budget",),
                default_budget=0.1,
            ),
        )
    )


def main() -> None:
    policy = build_policy()
    guard = PrivacyGuardV0(
        exact_verifier=LeakageVerifier(policy),
        semantic_verifier=SemanticLeakageVerifierV0(policy),
        rewriter=TemplateAbstractionRewriter(policy),
    )

    private_context = "We are negotiating with Vendor B. Our budget ceiling is $2M."
    counterfactual = build_counterfactual_text(private_context, policy)
    print("Counterfactual context:")
    print(counterfactual.public_text)

    candidates = [
        "Our maximum budget is $2M.",
        "We cannot exceed 2 million dollars.",
        "We have a constrained budget and should negotiate scope.",
    ]
    for candidate in candidates:
        decision = guard.protect(candidate)
        print("\nCandidate:", candidate)
        print("Accepted:", decision.accepted)
        print("Reasons:", decision.reasons)
        print("Final:", decision.final_text)
        for finding in decision.semantic_verification.findings:
            print(
                f"  semantic finding: {finding.fact_id} "
                f"{finding.label} score={finding.score} evidence={finding.evidence}"
            )


if __name__ == "__main__":
    main()

