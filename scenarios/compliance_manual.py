"""
Memex OS-Agent Benchmark — Compliance Manual (Enterprise Intranet Corpus).

A searchable text corpus of fragmented compliance rules that the agent can
discover via ``search_compliance_manual`` and inject into its kernel via
``update_system_prompt``.

Each rule has:
  - rule_id:      Unique identifier (e.g., "RULE-301")
  - title:        Short descriptive title
  - category:     Typology category (structuring, layering, trade_based_ml, general)
  - keywords:     Search terms for keyword matching
  - text:         The full rule text to be injected into the system prompt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ComplianceRule:
    """A single enterprise compliance policy rule."""
    rule_id: str
    title: str
    category: str
    keywords: List[str]
    text: str


# The Corpus — ~15 rules covering the three typologies + general AML

COMPLIANCE_RULES: List[ComplianceRule] = [
    # ---- Structuring -------------------------------------------------- #
    ComplianceRule(
        rule_id="RULE-101",
        title="Currency Transaction Report Threshold",
        category="structuring",
        keywords=["ctr", "threshold", "10000", "reporting", "currency"],
        text=(
            "RULE-101: All cash transactions exceeding $10,000 USD require a Currency "
            "Transaction Report (CTR). Splitting deposits to avoid this threshold "
            "constitutes structuring (31 USC §5324)."
        ),
    ),
    ComplianceRule(
        rule_id="RULE-102",
        title="Structuring Indicators",
        category="structuring",
        keywords=["structuring", "smurfing", "sub_threshold", "cash", "deposits"],
        text=(
            "RULE-102: Key structuring indicators: (a) Multiple cash deposits in amounts "
            "between $3,000-$9,999 within a 7-day window, (b) same branch or nearby "
            "branches, (c) no business justification for cash activity, (d) aggregate "
            "total exceeds $10,000. File SAR-8 if ≥3 indicators present."
        ),
    ),
    ComplianceRule(
        rule_id="RULE-103",
        title="Cash-Intensive Business Exception",
        category="structuring",
        keywords=["cash", "business", "occupation", "justification", "expected"],
        text=(
            "RULE-103: Cash-intensive businesses (restaurants, laundromats, parking lots) "
            "may have legitimate high cash volumes. Verify declared occupation against "
            "expected monthly cash activity before flagging. Non-cash occupations "
            "(clerks, IT workers) with high cash deposits are red flags."
        ),
    ),

    # ---- Layering ----------------------------------------------------- #
    ComplianceRule(
        rule_id="RULE-201",
        title="Rapid Fund Dispersal",
        category="layering",
        keywords=["fan_out", "dispersal", "rapid", "shell", "layering", "multiple"],
        text=(
            "RULE-201: Rapid fan-out — funds received and immediately split to 3+ "
            "entities within 24-48 hours — is a primary layering indicator. Especially "
            "suspicious when recipients are newly incorporated or share a registered "
            "address."
        ),
    ),
    ComplianceRule(
        rule_id="RULE-202",
        title="Shell Company Red Flags",
        category="layering",
        keywords=["shell", "company", "nominee", "incorporation", "registered_address"],
        text=(
            "RULE-202: Shell company indicators: (a) Registered within the last 12 "
            "months, (b) no employees or physical operations, (c) shared registered "
            "address with other suspects, (d) nominee directors. Cross-reference "
            "with beneficial ownership records."
        ),
    ),
    ComplianceRule(
        rule_id="RULE-203",
        title="Politically Exposed Persons",
        category="layering",
        keywords=["pep", "politically_exposed", "person", "enhanced", "due_diligence"],
        text=(
            "RULE-203: Politically Exposed Persons (PEPs) and their associates require "
            "Enhanced Due Diligence (EDD). Any transaction chain touching a PEP — even "
            "indirectly through shell entities — must be escalated. PEP connections "
            "increase layering risk significantly."
        ),
    ),
    ComplianceRule(
        rule_id="RULE-204",
        title="Shared Address Networks",
        category="layering",
        keywords=["shared", "address", "network", "common", "registered"],
        text=(
            "RULE-204: When network analysis reveals multiple entities sharing a "
            "registered address, this is a strong indicator of coordinated layering. "
            "Trace beneficial ownership of ALL entities at the shared address."
        ),
    ),

    # ---- Trade-Based ML ----------------------------------------------- #
    ComplianceRule(
        rule_id="RULE-301",
        title="Over/Under Invoicing",
        category="trade_based_ml",
        keywords=["invoice", "over_invoicing", "under_invoicing", "price", "trade", "market"],
        text=(
            "RULE-301: Trade-Based Money Laundering (TBML) detection — compare invoiced "
            "unit prices against market benchmarks. Deviations >30% above (over-invoicing) "
            "or >30% below (under-invoicing) market value are red flags, especially when "
            "combined with FATF-jurisdiction counterparties."
        ),
    ),
    ComplianceRule(
        rule_id="RULE-302",
        title="FATF High-Risk Jurisdictions",
        category="trade_based_ml",
        keywords=["fatf", "jurisdiction", "high_risk", "monitored", "grey_list"],
        text=(
            "RULE-302: Transactions involving FATF grey-list jurisdictions require "
            "enhanced scrutiny. Current high-risk jurisdictions include: Myanmar, "
            "Iran, DPRK, Syria. Monitored: South Sudan, Yemen, Nigeria (partial). "
            "Any trade flow touching these jurisdictions warrants SAR consideration."
        ),
    ),
    ComplianceRule(
        rule_id="RULE-303",
        title="Beneficial Ownership in Trade",
        category="trade_based_ml",
        keywords=["beneficial_owner", "ownership", "related_party", "trade", "connection"],
        text=(
            "RULE-303: When the buyer and seller in a trade transaction share a "
            "beneficial owner (directly or through intermediaries), over/under-invoicing "
            "is a method to transfer value without detection. Always cross-reference "
            "ownership records between trade counterparties."
        ),
    ),
    ComplianceRule(
        rule_id="RULE-304",
        title="Reversed and Amended Trade Transactions",
        category="trade_based_ml",
        keywords=["reversed", "reversal", "amended", "corrected", "transaction"],
        text=(
            "RULE-304: Reversed, amended, or 'corrected' trade transactions — especially "
            "when the reversal occurs after payment — are used to disguise the true "
            "nature of value transfers. Flag any invoice reversal followed by a "
            "re-issuance at a different price."
        ),
    ),

    # ---- General AML -------------------------------------------------- #
    ComplianceRule(
        rule_id="RULE-401",
        title="SAR Filing Deadline",
        category="general",
        keywords=["sar", "deadline", "filing", "days", "suspicious"],
        text=(
            "RULE-401: A SAR must be filed within 30 calendar days of initial detection "
            "of suspicious activity. If the subject is unknown, an additional 30 days "
            "is permitted for identification (60 days total)."
        ),
    ),
    ComplianceRule(
        rule_id="RULE-402",
        title="SAR Narrative Requirements",
        category="general",
        keywords=["sar", "narrative", "requirements", "evidence", "documentation"],
        text=(
            "RULE-402: SAR narrative must include: (1) the 5 W's — who, what, when, "
            "where, why; (2) specific transaction IDs and amounts; (3) identified "
            "typology; (4) all entities involved; (5) supporting evidence from KYC, "
            "watchlist, network, and source-of-funds checks."
        ),
    ),
    ComplianceRule(
        rule_id="RULE-403",
        title="Wire Transfer Documentation",
        category="general",
        keywords=["wire", "transfer", "trace", "international", "documentation"],
        text=(
            "RULE-403: International wire transfers >$3,000 require full originator "
            "and beneficiary information under the Travel Rule. Request wire trace "
            "documentation for any cross-border transaction lacking complete party "
            "identification."
        ),
    ),
    ComplianceRule(
        rule_id="RULE-404",
        title="Source of Funds Verification",
        category="general",
        keywords=["source", "funds", "verification", "documentation", "unexplained"],
        text=(
            "RULE-404: Source of funds documentation is required for: (a) any single "
            "transaction >$25,000, (b) aggregate deposits >$50,000 in a 30-day period, "
            "(c) any transaction flagged by automated monitoring. Unverified sources "
            "are a critical red flag."
        ),
    ),
]


# Search Engine

def search_compliance_manual(
    query: str,
    max_results: int = 3,
    category_filter: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Keyword search over the compliance manual corpus.

    Args:
        query: Free-text search query.
        max_results: Maximum rules to return.
        category_filter: Optional category restriction.

    Returns:
        List of dicts with rule_id, title, category, and text.
    """
    query_tokens = set(query.lower().replace("-", "_").replace(" ", "_").split("_"))
    query_tokens = {t for t in query_tokens if len(t) > 2}

    if not query_tokens:
        query_tokens = set(query.lower().split())

    scored: List[tuple[float, ComplianceRule]] = []

    for rule in COMPLIANCE_RULES:
        if category_filter and rule.category != category_filter:
            continue

        # Score = number of query tokens matching keywords + title + category
        searchable = set(rule.keywords) | set(rule.title.lower().split()) | {rule.category}
        # Also add tokens from the rule text for broader matching
        text_tokens = set(rule.text.lower().replace("-", "_").replace(",", "").split())

        # Primary: keyword hits (weight 3x)
        keyword_hits = len(query_tokens & set(rule.keywords)) * 3.0
        # Secondary: title hits
        title_hits = len(query_tokens & set(rule.title.lower().split()))
        # Tertiary: text body hits
        text_hits = len(query_tokens & text_tokens) * 0.5

        score = keyword_hits + title_hits + text_hits
        if score > 0:
            scored.append((score, rule))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "rule_id": rule.rule_id,
            "title": rule.title,
            "category": rule.category,
            "text": rule.text,
        }
        for _, rule in scored[:max_results]
    ]
