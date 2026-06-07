from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_runtime import (  # noqa: E402
    CandidateAction,
    ForbiddenStringConstraint,
    LeakageVerifier,
    PrivacyAccountant,
    PrivacyLogitProcessor,
    PrivacyPolicy,
    PrivacyRuntime,
    ProtectedFact,
    SimpleVocabulary,
)


def build_policy() -> PrivacyPolicy:
    return PrivacyPolicy(
        facts=(
            ProtectedFact(
                fact_id="s_budget",
                fact="budget ceiling is $2M",
                surface_forms=("$2M", "2M", "two million", "two million dollars"),
                semantic_hints=("budget ceiling", "maximum budget", "cannot exceed"),
                allowed_abstractions=("constrained budget", "limited budget"),
                default_budget=0.15,
                channel_budgets={"third_party_message": 0.10, "final": 0.25},
            ),
        )
    )


def show_token_mask(policy: PrivacyPolicy) -> None:
    vocabulary = SimpleVocabulary.from_tokens(
        [
            "$2M",
            "two million",
            "a constrained budget",
            "more flexibility",
            ".",
        ]
    )
    processor = PrivacyLogitProcessor(
        vocabulary=vocabulary,
        constraints=(ForbiddenStringConstraint(policy.all_forbidden_strings()),),
    )
    generated = "Our maximum budget is "
    decision = processor.blocked_tokens(generated)

    print("Token mask demo")
    for token_id in sorted(decision.blocked_token_ids):
        print(f"  blocked token {token_id}: {vocabulary.token_text(token_id)!r}")
        for reason in decision.reasons[token_id]:
            print(f"    reason: {reason}")


def show_runtime_decision(policy: PrivacyPolicy) -> None:
    runtime = PrivacyRuntime(
        verifier=LeakageVerifier(policy),
        accountant=PrivacyAccountant(policy),
    )
    candidates = [
        CandidateAction(
            text="Our maximum budget is $2M.",
            utility=0.98,
            channel="third_party_message",
            recipient="vendor_b",
        ),
        CandidateAction(
            text="We have a constrained budget, but we can discuss scope.",
            utility=0.88,
            channel="third_party_message",
            recipient="vendor_b",
        ),
        CandidateAction(
            text="I cannot disclose our internal budget ceiling.",
            utility=0.62,
            channel="third_party_message",
            recipient="vendor_b",
        ),
    ]

    decision = runtime.choose(candidates)

    print("\nRuntime decision demo")
    for rejected in decision.rejected:
        print(f"  rejected: {rejected.candidate.text!r}")
        print(f"    reason: {rejected.reason}")
        for finding in rejected.verification.findings:
            print(f"    finding: {finding}")

    if decision.selected is None:
        print("  no valid candidate selected")
        return

    print(f"  selected: {decision.selected.text!r}")
    print(f"  privacy cost: {dict(decision.verification.costs)}")
    print(
        "  remaining budget:",
        runtime.accountant.remaining("s_budget", "third_party_message"),
    )


def main() -> None:
    policy = build_policy()
    show_token_mask(policy)
    show_runtime_decision(policy)


if __name__ == "__main__":
    main()

