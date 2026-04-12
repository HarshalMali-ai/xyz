"""Rich task catalog for the RAG pipeline debugger environment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

ACTION_SCHEMA: dict[str, dict[str, Any]] = {
    "configure": {
        "description": "Update one or more retrieval or indexing knobs before the next simulation step.",
        "payload_fields": {
            "chunk_size": {"type": "integer", "required": False, "description": "Target chunk size in estimated tokens."},
            "chunk_overlap": {"type": "integer", "required": False, "description": "Overlap between adjacent chunks in tokens."},
            "top_k": {"type": "integer", "required": False, "description": "Number of chunks retrieved before answer generation."},
            "embedding_model": {"type": "string", "required": False, "description": "Embedding model used to build the vector index."},
            "query_embedding_model": {
                "type": "string",
                "required": False,
                "description": "Embedding model used for live queries.",
            },
            "rerank_enabled": {"type": "boolean", "required": False, "description": "Whether a reranker is applied after retrieval."},
            "max_context_tokens": {
                "type": "integer",
                "required": False,
                "description": "Budget available to the answering model after retrieval.",
            },
        },
    },
    "reindex": {
        "description": "Rebuild the index after chunking or embedding changes.",
        "payload_fields": {},
    },
    "submit": {
        "description": "Submit the final pipeline state for grading.",
        "payload_fields": {},
    },
    "request_hint": {
        "description": "Ask for the next hint. This helps but slightly reduces the reward.",
        "payload_fields": {},
    },
}


def _doc(doc_id: str, title: str, role: str, tokens: int, summary: str) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": title,
        "role": role,
        "tokens": tokens,
        "summary": summary,
    }


TASK_LIBRARY: dict[str, dict[str, Any]] = {
    "easy_chunk_alignment": {
        "difficulty": "easy",
        "title": "Fix Oversized Chunks For Support Runbooks",
        "description": "The retriever is indexing long support runbooks as giant chunks, so answers cite vague paragraphs instead of the exact procedural steps.",
        "operator_story": "A support engineer is trying to answer how to rotate an API key without downtime.",
        "business_impact": "Wrong retrieval slows support replies and increases escalations during customer incidents.",
        "user_query": "What are the exact steps to rotate an API key without breaking active webhook deliveries?",
        "default_config": {
            "chunk_size": 2200,
            "chunk_overlap": 0,
            "embedding_model": "text-embedding-3-small",
            "query_embedding_model": "text-embedding-3-small",
            "top_k": 6,
            "rerank_enabled": True,
            "reindex_completed": False,
            "max_context_tokens": 5500,
        },
        "target_config": {"chunk_size": 450},
        "reindex_required": True,
        "overflow_sensitive": False,
        "max_steps": 14,
        "ideal_retrieval_ids": ["doc_runbook_api_rotation", "doc_webhook_replay_guardrails", "doc_release_notes"],
        "hints": [
            "The answer exists, but the current chunks are too broad to isolate the step-by-step procedure.",
            "Support runbooks in this scenario are around 400-500 tokens each after chunking.",
            "After changing chunking, you must rebuild the index before the retriever improves.",
        ],
        "acceptance_criteria": [
            "Chunks align with procedure-sized sections.",
            "The index is rebuilt after the chunking change.",
            "The retrieved preview starts with the API rotation runbook instead of general release notes.",
        ],
        "corpus": [
            _doc("doc_runbook_api_rotation", "API Key Rotation Runbook", "gold", 480, "Step-by-step rotation procedure and rollback guidance."),
            _doc("doc_webhook_replay_guardrails", "Webhook Replay Guardrails", "support", 430, "How to keep webhooks flowing during key swaps."),
            _doc("doc_release_notes", "Quarterly Platform Release Notes", "support", 310, "Mentions API keys, but mostly broad platform updates."),
            _doc("doc_marketing_launch", "Marketing Launch Checklist", "noise", 260, "Unrelated launch checklist with generic approval steps."),
            _doc("doc_billing_faq", "Billing FAQ", "noise", 220, "Common invoicing answers with no key rotation steps."),
        ],
    },
    "easy_chunk_overlap": {
        "difficulty": "easy",
        "title": "Restore Overlap For Split Incident Playbooks",
        "description": "Chunk boundaries cut multi-step playbooks in half, so the agent retrieves incomplete remediations when an instruction spills across chunk edges.",
        "operator_story": "The on-call team needs the exact rollback sequence for a failed feature flag rollout.",
        "business_impact": "Missing one procedural step can turn a rollback into a customer-facing outage.",
        "user_query": "What is the rollback sequence when a feature flag rollout causes elevated latency in one region?",
        "default_config": {
            "chunk_size": 500,
            "chunk_overlap": 0,
            "embedding_model": "text-embedding-3-small",
            "query_embedding_model": "text-embedding-3-small",
            "top_k": 5,
            "rerank_enabled": True,
            "reindex_completed": False,
            "max_context_tokens": 4500,
        },
        "target_config": {"chunk_overlap": 80},
        "reindex_required": True,
        "overflow_sensitive": False,
        "max_steps": 14,
        "ideal_retrieval_ids": ["doc_flag_rollback_playbook", "doc_latency_triage", "doc_incident_roles"],
        "hints": [
            "The correct document is already in the corpus, but the right sentence straddles a chunk boundary.",
            "A modest overlap is enough; you do not need to shrink chunk size further.",
            "Because chunk boundaries changed, rebuild the index after updating overlap.",
        ],
        "acceptance_criteria": [
            "Chunk overlap preserves instructions that span chunk boundaries.",
            "The rollout rollback playbook becomes the top retrieval result.",
            "The index reflects the new overlap setting.",
        ],
        "corpus": [
            _doc("doc_flag_rollback_playbook", "Feature Flag Rollback Playbook", "gold", 560, "Regional rollback sequence, guardrail checks, and re-enable conditions."),
            _doc("doc_latency_triage", "Latency Triage Guide", "support", 360, "Symptom checklist used before choosing rollback versus retry."),
            _doc("doc_incident_roles", "Incident Roles Matrix", "support", 300, "Who owns comms, who owns rollback, and approval order."),
            _doc("doc_design_review", "Design Review Notes", "noise", 280, "Architecture tradeoffs unrelated to rollback execution."),
            _doc("doc_sales_enablement", "Sales Enablement Deck", "noise", 210, "Slide content with no operational guidance."),
        ],
    },
    "easy_top_k_budget": {
        "difficulty": "easy",
        "title": "Trim Top-K For Short FAQ Answers",
        "description": "The retriever pulls too many FAQ fragments for short answers, burying the one exact policy answer beneath repeated near-matches.",
        "operator_story": "A customer success agent needs the exact refund eligibility window from the knowledge base.",
        "business_impact": "Over-retrieval increases hallucinations and makes policy answers inconsistent across teams.",
        "user_query": "How many calendar days does a customer have to request a self-serve refund after renewal?",
        "default_config": {
            "chunk_size": 320,
            "chunk_overlap": 40,
            "embedding_model": "text-embedding-3-small",
            "query_embedding_model": "text-embedding-3-small",
            "top_k": 10,
            "rerank_enabled": True,
            "reindex_completed": True,
            "max_context_tokens": 2500,
        },
        "target_config": {"top_k": 4},
        "reindex_required": False,
        "overflow_sensitive": True,
        "max_steps": 12,
        "ideal_retrieval_ids": ["doc_refund_policy", "doc_billing_proration", "doc_exception_approvals"],
        "hints": [
            "The knowledge base chunks are already good. The problem is how many results you pass downstream.",
            "This question can be answered from a handful of high-quality chunks.",
            "If the context window clears, the refund policy should move to the top without a reindex.",
        ],
        "acceptance_criteria": [
            "Context no longer floods with low-value FAQ fragments.",
            "The refund policy document becomes the highest-value retrieval.",
            "No reindex is necessary for this task.",
        ],
        "corpus": [
            _doc("doc_refund_policy", "Refund Policy", "gold", 240, "Defines the exact refund window and exclusions after renewal."),
            _doc("doc_billing_proration", "Billing Proration Guide", "support", 210, "Explains when credits are applied instead of refunds."),
            _doc("doc_exception_approvals", "Exception Approval SOP", "support", 260, "Escalation path for out-of-policy refund requests."),
            _doc("doc_trial_faq", "Trial FAQ", "noise", 190, "Trial cancellation guidance, not renewal refunds."),
            _doc("doc_partner_terms", "Partner Terms", "noise", 260, "Channel-partner reimbursement rules."),
            _doc("doc_security_commitments", "Security Commitments", "noise", 230, "Compliance language with no billing policy."),
        ],
    },
    "medium_embedding_migration": {
        "difficulty": "medium",
        "title": "Complete A Partial Embedding Migration",
        "description": "The vector index still uses an older embedding family while live queries use the new model, causing semantic drift after a migration.",
        "operator_story": "The AI support bot stopped finding the right runbooks after an embedding upgrade.",
        "business_impact": "Migration regressions create silent quality drops that are hard for teams to detect without tracing retrieval.",
        "user_query": "Which runbook should I follow when webhook signature verification fails only for rotated secrets?",
        "default_config": {
            "chunk_size": 450,
            "chunk_overlap": 70,
            "embedding_model": "text-embedding-ada-002",
            "query_embedding_model": "text-embedding-3-small",
            "top_k": 6,
            "rerank_enabled": True,
            "reindex_completed": True,
            "max_context_tokens": 4200,
        },
        "target_config": {
            "embedding_model": "text-embedding-3-small",
            "query_embedding_model": "text-embedding-3-small",
        },
        "reindex_required": True,
        "overflow_sensitive": False,
        "max_steps": 16,
        "ideal_retrieval_ids": ["doc_signature_rotation", "doc_secret_rollout_order", "doc_audit_logging"],
        "hints": [
            "Query embeddings already use the desired model family.",
            "Index and query vectors need to live in the same representation space.",
            "Changing the index model without rebuilding will not improve retrieval.",
        ],
        "acceptance_criteria": [
            "Index and query embeddings match.",
            "The index is rebuilt after the model change.",
            "The signature rotation runbook becomes the first retrieved document.",
        ],
        "corpus": [
            _doc("doc_signature_rotation", "Webhook Signature Rotation", "gold", 420, "Secret rotation ordering and validation checks."),
            _doc("doc_secret_rollout_order", "Secret Rollout Order", "support", 340, "How to rotate secrets without invalidating in-flight traffic."),
            _doc("doc_audit_logging", "Audit Logging For Rotations", "support", 310, "Evidence and logging requirements during secret changes."),
            _doc("doc_dashboard_tips", "Dashboard Tips", "noise", 200, "UI shortcuts unrelated to webhook verification."),
            _doc("doc_invoice_exports", "Invoice Export Guide", "noise", 250, "Billing export workflow with no secret handling."),
        ],
    },
    "medium_multilingual_embeddings": {
        "difficulty": "medium",
        "title": "Fix Cross-Lingual Retrieval For Regional Support",
        "description": "Regional Hindi and English troubleshooting articles stopped aligning after the team switched only the query side to a multilingual embedding model.",
        "operator_story": "Regional support needs to answer the same login issue from bilingual documentation.",
        "business_impact": "Cross-lingual retrieval failures create uneven customer experience across markets.",
        "user_query": "Hindi help articles say the OTP delay is caused by carrier throttling. Which remediation steps should support send first?",
        "default_config": {
            "chunk_size": 420,
            "chunk_overlap": 60,
            "embedding_model": "text-embedding-3-small",
            "query_embedding_model": "text-embedding-3-large",
            "top_k": 5,
            "rerank_enabled": True,
            "reindex_completed": True,
            "max_context_tokens": 3600,
        },
        "target_config": {
            "embedding_model": "text-embedding-3-large",
            "query_embedding_model": "text-embedding-3-large",
        },
        "reindex_required": True,
        "overflow_sensitive": False,
        "max_steps": 16,
        "ideal_retrieval_ids": ["doc_hindi_otp_delay", "doc_carrier_throttling_playbook", "doc_english_otp_delay"],
        "hints": [
            "The query side already points to the multilingual model.",
            "Regional documents only align when the index was built with the same multilingual model family.",
            "Rebuild the index once the embedding model is updated.",
        ],
        "acceptance_criteria": [
            "Regional and English OTP guides are retrieved together.",
            "Embedding families match across indexing and query time.",
            "A reindex occurs after the model change.",
        ],
        "corpus": [
            _doc("doc_hindi_otp_delay", "Hindi OTP Delay Guide", "gold", 350, "Hindi remediation steps for OTP delays and carrier throttling."),
            _doc("doc_carrier_throttling_playbook", "Carrier Throttling Playbook", "support", 380, "Escalation path and messaging template for telecom delays."),
            _doc("doc_english_otp_delay", "English OTP Delay Guide", "support", 320, "English troubleshooting variant of the same OTP issue."),
            _doc("doc_marketing_holiday", "Holiday Campaign Notes", "noise", 260, "Campaign calendar with no OTP troubleshooting."),
            _doc("doc_app_theme_tokens", "App Theme Tokens", "noise", 220, "Design system tokens unrelated to support content."),
        ],
    },
    "medium_rerank_precision": {
        "difficulty": "medium",
        "title": "Re-Enable Reranking After Hybrid Retrieval",
        "description": "Hybrid retrieval is pulling many loosely relevant policy pages ahead of the incident playbook because reranking was disabled during a rollout.",
        "operator_story": "The incident bot needs the exact runbook for repeated 429s from a downstream dependency.",
        "business_impact": "Without reranking, support agents spend extra minutes scanning the wrong pages during incidents.",
        "user_query": "Which mitigation steps should we apply first when our partner API starts returning sustained 429s?",
        "default_config": {
            "chunk_size": 420,
            "chunk_overlap": 60,
            "embedding_model": "text-embedding-3-small",
            "query_embedding_model": "text-embedding-3-small",
            "top_k": 7,
            "rerank_enabled": False,
            "reindex_completed": True,
            "max_context_tokens": 4200,
        },
        "target_config": {"rerank_enabled": True},
        "reindex_required": False,
        "overflow_sensitive": False,
        "max_steps": 15,
        "ideal_retrieval_ids": ["doc_429_playbook", "doc_partner_backoff", "doc_rate_limit_dashboard"],
        "hints": [
            "The right documents are present in the top-k set, but they are ordered badly.",
            "No embedding migration is needed here.",
            "A reranker should promote the operational playbook ahead of general policy pages.",
        ],
        "acceptance_criteria": [
            "The 429 mitigation playbook becomes the top retrieval result.",
            "General policy pages are pushed below operational docs.",
            "No reindex is required.",
        ],
        "corpus": [
            _doc("doc_429_playbook", "Partner 429 Mitigation Playbook", "gold", 390, "Operational steps for throttling, queueing, and customer communication."),
            _doc("doc_partner_backoff", "Partner Backoff Strategy", "support", 330, "Retry windows and safe concurrency reductions."),
            _doc("doc_rate_limit_dashboard", "Rate Limit Dashboard Guide", "support", 280, "Which panels confirm the issue and recovery."),
            _doc("doc_api_terms", "Partner API Terms", "noise", 360, "General contractual terms mentioning rate limits."),
            _doc("doc_governance_review", "Governance Review Notes", "noise", 260, "Long-form policy notes unrelated to incident response."),
        ],
    },
    "hard_context_overflow": {
        "difficulty": "hard",
        "title": "Recover A Long-Context Overflow Regression",
        "description": "A long-context answer chain is flooding the model with too many chunks after a top-k increase and reranker rollback.",
        "operator_story": "The assistant must answer a customer escalation using only the most relevant postmortem and playbook excerpts.",
        "business_impact": "Context overflow causes slow, contradictory answers exactly when leadership is watching incident comms.",
        "user_query": "What customer-safe workaround should we communicate while the cross-region write path is degraded?",
        "default_config": {
            "chunk_size": 650,
            "chunk_overlap": 60,
            "embedding_model": "text-embedding-3-small",
            "query_embedding_model": "text-embedding-3-small",
            "top_k": 18,
            "rerank_enabled": False,
            "reindex_completed": True,
            "max_context_tokens": 1800,
        },
        "target_config": {"top_k": 3, "rerank_enabled": True},
        "reindex_required": False,
        "overflow_sensitive": True,
        "max_steps": 18,
        "ideal_retrieval_ids": ["doc_write_path_workaround", "doc_customer_comms_template", "doc_failover_postmortem"],
        "hints": [
            "The issue is not missing data. It is too much low-value context competing for the model budget.",
            "Reduce the number of retrieved chunks and promote the best ones.",
            "A reranker is required to keep the postmortem and comms template above noisy context.",
        ],
        "acceptance_criteria": [
            "Context overflow clears.",
            "Only the highest-value customer workaround material remains in retrieval.",
            "The top retrieval order reflects operational priority rather than raw lexical overlap.",
        ],
        "corpus": [
            _doc("doc_write_path_workaround", "Cross-Region Write Path Workaround", "gold", 420, "Short-term customer-safe workaround during write degradation."),
            _doc("doc_customer_comms_template", "Customer Comms Template", "support", 340, "Approved language for degraded-service updates."),
            _doc("doc_failover_postmortem", "Failover Postmortem Excerpts", "support", 520, "Lessons learned and safe guardrails for temporary routing."),
            _doc("doc_full_incident_timeline", "Full Incident Timeline", "noise", 900, "Long narrative timeline that floods context."),
            _doc("doc_dependency_matrix", "Dependency Matrix", "noise", 760, "Large systems map with little direct answer value."),
            _doc("doc_audit_controls", "Audit Controls", "noise", 500, "Controls language irrelevant to the workaround answer."),
        ],
    },
    "hard_release_migration": {
        "difficulty": "hard",
        "title": "Repair A Release Notes Retrieval Stack After Index Drift",
        "description": "The team changed embeddings during a release-notes migration, forgot to rebuild the index, and also left retrieval too broad for customer-facing release explanations.",
        "operator_story": "Support needs the exact migration note for a deprecated webhook behavior before a premium customer call.",
        "business_impact": "Wrong release guidance erodes trust during upgrades and increases avoidable support tickets.",
        "user_query": "Which release note explains the webhook retry policy change introduced in the last platform upgrade?",
        "default_config": {
            "chunk_size": 520,
            "chunk_overlap": 80,
            "embedding_model": "text-embedding-ada-002",
            "query_embedding_model": "text-embedding-3-small",
            "top_k": 10,
            "rerank_enabled": False,
            "reindex_completed": True,
            "max_context_tokens": 2200,
        },
        "target_config": {
            "embedding_model": "text-embedding-3-small",
            "query_embedding_model": "text-embedding-3-small",
            "top_k": 4,
            "rerank_enabled": True,
        },
        "reindex_required": True,
        "overflow_sensitive": True,
        "max_steps": 18,
        "ideal_retrieval_ids": ["doc_release_retry_change", "doc_migration_checklist", "doc_webhook_versioning"],
        "hints": [
            "There is more than one failure mode here: semantic drift and context sprawl.",
            "The correct release note only surfaces after embeddings align and the index is rebuilt.",
            "You also need to reduce retrieval breadth so the change log stays within context budget.",
        ],
        "acceptance_criteria": [
            "Embeddings are aligned and the index is rebuilt.",
            "Top-k is reduced enough to keep the answer chain inside budget.",
            "The webhook retry policy change note appears before generic migration docs.",
        ],
        "corpus": [
            _doc("doc_release_retry_change", "Release Note: Webhook Retry Policy", "gold", 360, "Explains the retry behavior change and migration caveats."),
            _doc("doc_migration_checklist", "Upgrade Migration Checklist", "support", 480, "Checklist for rollout, validation, and rollback."),
            _doc("doc_webhook_versioning", "Webhook Versioning Guide", "support", 340, "Version compatibility and customer messaging."),
            _doc("doc_full_release_digest", "Full Release Digest", "noise", 900, "Very long digest with many unrelated sections."),
            _doc("doc_pricing_update", "Pricing Update Notes", "noise", 420, "Pricing changes unrelated to webhook retries."),
            _doc("doc_roadmap_brainstorm", "Roadmap Brainstorm", "noise", 610, "Internal planning notes with high lexical overlap."),
        ],
    },
    "hard_multiknob_repair": {
        "difficulty": "hard",
        "title": "Stabilize A Multi-Knob Retrieval Failure In Postmortem Search",
        "description": "Postmortem search is failing because chunks are too large, overlap is missing, reranking is off, and the answer chain is over-fetching evidence.",
        "operator_story": "An engineering leader wants the exact remediation commitments from a prior postmortem before an executive review.",
        "business_impact": "When postmortems are hard to retrieve, teams repeat old incidents and lose trust in their knowledge system.",
        "user_query": "What remediation commitments were agreed after the duplicate charge postmortem, and which owner was assigned to each action?",
        "default_config": {
            "chunk_size": 2400,
            "chunk_overlap": 0,
            "embedding_model": "text-embedding-3-small",
            "query_embedding_model": "text-embedding-3-small",
            "top_k": 8,
            "rerank_enabled": False,
            "reindex_completed": False,
            "max_context_tokens": 2100,
        },
        "target_config": {
            "chunk_size": 700,
            "chunk_overlap": 120,
            "top_k": 3,
            "rerank_enabled": True,
        },
        "reindex_required": True,
        "overflow_sensitive": True,
        "max_steps": 20,
        "ideal_retrieval_ids": ["doc_duplicate_charge_postmortem", "doc_commitment_tracker", "doc_owner_handbook"],
        "hints": [
            "This task is intentionally multi-factor. Fixing only one knob will not recover the right postmortem evidence.",
            "Postmortems contain long sections that need better chunking and overlap before reranking helps.",
            "After the chunking changes, rebuild the index and then reduce top-k to stay within budget.",
        ],
        "acceptance_criteria": [
            "Chunks become small enough to isolate commitments and owners.",
            "Overlap preserves action items that span section boundaries.",
            "Reranking and lower top-k keep only the most relevant postmortem evidence in context.",
        ],
        "corpus": [
            _doc("doc_duplicate_charge_postmortem", "Duplicate Charge Postmortem", "gold", 840, "Incident narrative, remediation commitments, and owner mapping."),
            _doc("doc_commitment_tracker", "Remediation Commitment Tracker", "support", 460, "Follow-up owners and due dates for agreed fixes."),
            _doc("doc_owner_handbook", "Incident Owner Handbook", "support", 390, "Responsibilities for follow-through after a major incident."),
            _doc("doc_full_audit_appendix", "Audit Appendix", "noise", 950, "Very long appendix that bloats context without answering the query."),
            _doc("doc_finance_controls", "Finance Controls Matrix", "noise", 640, "Chargeback controls with no postmortem commitments."),
            _doc("doc_org_announcements", "Org Announcements", "noise", 500, "Company updates unrelated to the incident."),
        ],
    },
}

TASK_ALIASES = {
    "task_easy": "easy_chunk_alignment",
    "task_medium": "medium_embedding_migration",
    "task_hard": "hard_context_overflow",
}

FLAGSHIP_TASKS = (
    "easy_chunk_alignment",
    "medium_embedding_migration",
    "hard_context_overflow",
)


def resolve_task_id(task_id: str | None) -> str:
    if not task_id:
        return FLAGSHIP_TASKS[0]
    if task_id in TASK_ALIASES:
        return TASK_ALIASES[task_id]
    if task_id in TASK_LIBRARY:
        return task_id
    return FLAGSHIP_TASKS[0]


def get_task_spec(task_id: str | None) -> dict[str, Any]:
    resolved = resolve_task_id(task_id)
    spec = deepcopy(TASK_LIBRARY[resolved])
    spec["id"] = resolved
    return spec


def list_task_specs() -> list[dict[str, Any]]:
    return [get_task_spec(task_id) for task_id in TASK_LIBRARY]


def flagship_task_ids() -> tuple[str, ...]:
    return FLAGSHIP_TASKS


def list_tasks_payload() -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for task_id in TASK_LIBRARY:
        spec = get_task_spec(task_id)
        payload.append(
            {
                "id": task_id,
                "difficulty": spec["difficulty"],
                "title": spec["title"],
                "description": spec["description"],
                "grader": True,
                "grader_meta": {"type": "programmatic", "endpoint": f"/grade/{task_id}", "method": "GET"},
                "grader_endpoint": f"/grade/{task_id}",
                "grader_method": "GET",
                "action_schema": deepcopy(ACTION_SCHEMA),
                "business_impact": spec["business_impact"],
                "user_query": spec["user_query"],
                "acceptance_criteria": list(spec["acceptance_criteria"]),
                "reindex_required": bool(spec["reindex_required"]),
                "overflow_sensitive": bool(spec["overflow_sensitive"]),
            }
        )
    return payload
