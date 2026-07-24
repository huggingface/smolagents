# Contract Finder Agent_v2 — Code Review Report

**Review Scope:** Full implementation review of all 15 ACTs against the Widget Overview specification and the `Book_Finder_2` reference agent (`.ecg/widgets/Book_Finder_2/EGPT_AI/utility_functions/agents/book_finder_agent/book_finder_agent.py`).
**Verdict: NOT PRODUCTION READY.** The codebase contains a **fatal runtime-breaking integration defect** (Finding #1) that will cause every standard-flow request to crash with `AttributeError` before any other issue even surfaces. Several ACT-mandated business-logic requirements (LLM timeout/sentinel handling, exact-match citation extraction, metadata exact-match filtering, route-level payload validation) are missing or only partially stubbed.

---

## Severity Legend
| Severity | Meaning |
|---|---|
| CRITICAL | Breaks runtime execution / core business logic is absent or wrong |
| HIGH | Feature present but materially incomplete vs. ACT spec |
| MEDIUM | Inconsistency, dead code, or minor spec deviation |
| LOW | Cosmetic / documentation only |

---

## 1. CRITICAL — Orphaned Method Files Never Wired Into `ContractFinderAgent` Class

**Files:** `agents/contract_finder_agent/contract_finder_agent_methods.py`, `agents/contract_finder_agent/contract_finder_agent_additional_methods.py`, `agents/contract_finder_agent/contract_finder_agent.py`

ACT 6 (`_retrieve_content`), ACT 7 (`_calculate_scores`, `ngram_search_with_weighted_reward`), and ACT 8 (`_assemble_context`) were implemented as **free-standing, top-level functions** (taking `self` as their first positional parameter) inside two separate files:
- `contract_finder_agent_methods.py` (lines 17, 112, 191, 306)
- `contract_finder_agent_additional_methods.py` (lines 16, 68, 117, 181) — an apparent near-duplicate of the same four functions, with a comment literally stating "To integrate: Copy these methods into the ContractFinderAgent class" (line 4), confirming the integration step was never completed.

However, `contract_finder_agent.py`'s `ContractFinderAgent` class (`run()` line 614, `fetch_contract_data()` line 708) calls `self._assemble_context(...)` and `self._retrieve_content(...)` directly. Verified via grep: **zero occurrences** of `def _retrieve_content`, `def _assemble_context`, `def _calculate_scores`, or `def ngram_search_with_weighted_reward` exist inside `contract_finder_agent.py`, and **zero imports/monkey-patch statements** reference either methods file from anywhere in the codebase.

**Impact:** Every call to `run()` in the standard Flow A path (the primary, documented use case of this widget) will raise `AttributeError: 'ContractFinderAgent' object has no attribute '_retrieve_content'` at `fetch_contract_data()` (line 708), and `run()` will similarly fail at `self._assemble_context(...)` (line 614) if that line is ever reached. **This makes the agent completely non-functional for Flow A out of the box.**

**Required Fix:** Move the four function bodies from both orphan files directly into the `ContractFinderAgent` class body in `contract_finder_agent.py` (as proper `def method_name(self, ...):` methods), delete the two now-redundant standalone files, and reconcile the two near-duplicate implementations (see Finding #2).

---

## 2. CRITICAL — Duplicate, Divergent Implementations of ACT 6/7/8 Logic

**Files:** `contract_finder_agent_methods.py` vs `contract_finder_agent_additional_methods.py`

Both files independently implement `_retrieve_content`, `ngram_search_with_weighted_reward`, `_calculate_scores`, and `_assemble_context` with **subtly different logic** (e.g., `_calculate_scores` in `_methods.py` spans lines 191-304 with more elaborate normalization branches vs. 117-178 in `_additional_methods.py`). Since neither is actually imported/attached to the class (Finding #1), this represents wasted/dead duplicate work that must be reconciled into a single canonical implementation, not simply both copy-pasted in.

---

## 3. CRITICAL — `FILTER_TABLE_MAPPING` Missing `'contract_content_chunks'` Key

**Files:** `contract_helper.py`, `contract_finder_agent_methods.py` (line 49), `contract_finder_agent_additional_methods.py` (line 29)

Both `_retrieve_content` implementations call `FILTER_TABLE_MAPPING.get('contract_content_chunks')` to look up the table/column mapping needed to build the `retrieve_final_chunks.sql` query (ACT 6, Step A4). However, `contract_helper.py`'s `FILTER_TABLE_MAPPING` dict (lines 44-141) only contains the 8 filter-type keys (`contract_details`, `product_type`, `party_details`, etc.) — **there is no `contract_content_chunks` entry**. Even if Finding #1 is fixed and these methods are wired into the class, `mapping` will always resolve to `None`, causing `_retrieve_content` to silently short-circuit and return `[]` on every call (line 30-31 / 49-51), meaning **no manuscript/clause chunks will ever be retrieved** — the entire Content Chunk Retrieval stage (ACT 6) is non-functional even after the wiring fix.

**Required Fix:** Add a dedicated `FilterDetails`/mapping entry for content-chunk retrieval (table: `contract_content_chunks` or equivalent per-clause chunk table, matching the reference's separate mapping used in `_retrieve_book_chunks`) to `FILTER_TABLE_MAPPING` or a new registry dict.

---

## 4. HIGH — ACT 9 (LLM Invocation & Response Handling) Core Requirements Absent

**File:** `contract_finder_agent.py`, `run()` method (lines 611-640)

ACT 9 explicitly mandates three behaviors, none of which are implemented:
1. **Synchronous invocation under a configured timeout** returning an error-sentinel string on timeout (`llm_invocation_timeout` field exists in `contract_finder_agent_setup.py` line 105-109 but is **never read or used** anywhere in `contract_finder_agent.py` — confirmed via grep, zero matches).
2. **No-context fallback template substitution**: `no_context_fallback_template` config field exists (setup.py line 121-126) but is **never referenced** in `run()`. Currently, when `formatted_context` is falsy, the code just falls through to `self.config.fallback_response` (line 638) — the generic fallback, not the dedicated no-context prompt-injection template the spec requires.
3. **Error-sentinel detection and replacement**: `llm_error_sentinel` and `llm_timeout_fallback_message` config fields exist (setup.py lines 111-120) but are **never referenced** in `run()`. The LLM result is used as-is (`post_processed_response = llm_result if llm_result else self.config.fallback_response`, line 633) with no sentinel-matching logic at all.

**Impact:** Timeout scenarios will hang indefinitely instead of returning gracefully (no timeout wrapper around `self.llm_config_tool.run(...)` at all — confirmed no `ThreadPoolExecutor`/`signal`/`concurrent.futures.TimeoutError` usage around the LLM call), and empty-context queries will get a generic apology instead of the spec-mandated conversational fallback.

---

## 5. HIGH — ACT 11 (Citation Generation) Is a Simplified Stub, Not the Specified Pipeline

**File:** `contract_finder_agent.py`, `_generate_citations()` (lines 1007-1058)

ACT 11 mandates: (a) parsing the LLM response via `_extract_sentences_with_citations` using numbered-sentence + markdown-link regex extraction, (b) invoking `self.citation_tool.run(citation_key=..., complete_content=..., cited_text=..., ...)` per citation via a `ThreadPoolExecutor` with `clone_for_thread()` for thread safety, and (c) a missing-document fallback via `fetch_missing_contracts_from_history.sql`.

The actual implementation:
- Does **not** call `self.citation_tool.run(...)` anywhere — confirmed via grep, `citation_tool` is only referenced in `setup()` (instantiation) and never invoked afterward.
- Does **not** implement `_extract_sentences_with_citations` — method does not exist in the file.
- Does **not** use `ThreadPoolExecutor`/`clone_for_thread()` for parallel citation processing (the import exists at line 8 but is unused throughout the entire file — confirmed via grep, zero call sites).
- Does **not** use the `fetch_missing_contracts_from_history.sql` query that was generated (confirmed via grep, zero references to this filename anywhere in `contract_finder_agent.py`).
- Instead, it builds one generic `citation_generator(...)` call per contract using a naive `chunk_text[:1000]` truncation — **no exact-match/verbatim-substring verification is performed at all**, defeating the entire stated purpose of Flow A6 ("generates exact-match citations").
- The dedicated "Semantic Citation Text Matcher" system prompt (analogous to reference's `book_finder_prompt.py`) that ACT 11 requires to be loaded during Step A2 setup was **never created** — confirmed via grep across the whole `agents/contract_finder_agent/` directory, no file or string containing "Semantic Citation Text Matcher" or a citation prompt constant exists.

**Impact:** Citations returned by the agent are unverified paraphrases/truncations, not the exact-match, source-verified citations that are the core value proposition described in the Widget Overview ("citations linked to exact contract clauses").

---

## 6. HIGH — Undefined Config Attributes Referenced at Runtime

**File:** `contract_finder_agent.py`, `run()` (lines 678, 682) and `_upsert_retrieved_documents()`/`_generate_citations()` (lines 890, 1029)

`self.config.smart_response_adjustment` and `self.config.max_contracts_to_return` are referenced but **never defined** as fields in `ContractFinderAgentSetup` (`contract_finder_agent_setup.py`, fully reviewed lines 1-127 — no such fields exist; the closest analogues defined are `contracts_per_result` and `max_show_more_contracts`, neither of which matches the referenced attribute names). Since `ContractFinderAgentSetup` is a Pydantic model, accessing an undeclared attribute will raise `AttributeError` at request time on every single call to `run()` (line 678 executes on every successful return path) and on every citation-generation/persistence call.

**Impact:** Combined with Finding #1, this is a second independent way the agent crashes on every request, even before reaching the LLM invocation stage.

---

## 7. HIGH — ACT 2 Payload Validation (Dedicated Route + 400 on Missing `user_query`) Not Implemented

**File:** `run.py`

ACT 2 requires a route (or the existing `/prediction` route) to explicitly validate the mandatory `user_query` field and return `jsonify({"error": "..."}), 400` before agent setup begins. The current `predict_api()` (lines 90-146) performs **no such validation** — it unpacks `data['agent_arguments']` directly into `rag_agent.run(...)` (line 86) with no pre-check, meaning a missing `user_query` will surface only deep inside `run()`'s own `ValueError` (line 559) and be caught by the generic outer `except Exception` in `predict_api()`, returning **HTTP 500**, not the spec-mandated **HTTP 400**.

---

## 8. HIGH — Metadata Exact-Match Filtering (`ContractFilterType`/`METADATA_EXACT_FILTERS`) Defined But Never Used

**Files:** `contract_helper.py` (lines 144-157), `contract_finder_agent.py`

`ContractFilterType` and `METADATA_EXACT_FILTERS` are defined per the Widget Overview's "Metadata Filtering Schemas" section and ACT 4's reference-file guidance, but **no code path in `contract_finder_agent.py` ever imports or reads `METADATA_EXACT_FILTERS`** (confirmed via grep — zero matches for `METADATA_EXACT_FILTERS`, `get_imprint_ids_from_metadata`, `get_isbns_using_metadata_filters`/equivalent, or `_build_metadata_filter_clause` anywhere in the agent). The generated `get_imprint_ids_from_metadata.sql`, `select_imprint_from_metadata_table.sql`, and `retrieve_imprints.sql` query files (present under `queries/`) are consequently **dead SQL** — never loaded into any executable code path.

---

## 9. MEDIUM — `_build_filter_query`'s "Associated Columns" Branch Selection Logic Is Never Actually Exercised

**File:** `contract_finder_agent.py`, `_build_filter_query()` (lines 384-459)

The branch condition `if "associated" in return_column.lower():` (line 409) checks the *return column name* for the substring "associated", but every `FilterDetails.return_column` value defined in `FILTER_TABLE_MAPPING` (`contract_id`, `party_id`, `territory_id`) contains no such substring — meaning the `filter_query_associated_cols.sql` branch is **permanently dead code** for the Contract Finder domain as currently mapped. The reference's equivalent registry (Book Finder) does have return columns like `"associated_isbns"` that trigger this branch; the Contract Finder's registry was ported without an equivalent "associated" mapping type, so this dual-template mechanism silently collapses to always using `filter_query_non_associated_cols.sql`.

---

## 10. MEDIUM — `run.py` Calls Undefined `get_llm_config()` Method

**File:** `run.py` (line 167), `contract_finder_agent.py`

The `/get_llm_config` route calls `AGENT_NAME().get_llm_config()`. No `get_llm_config` method (instance or inherited) is defined anywhere in `ContractFinderAgent`, `contract_finder_agent_methods.py`, or `contract_finder_agent_additional_methods.py` (confirmed via grep across all three files — zero matches). This route will raise `AttributeError` on every call unless it is inherited from `RecommendationAgentBase`/`AgentBase`, which was not verified as part of this widget's generated code (that base class is a read-only reference dependency, not part of the generated deliverable) — this should be confirmed or the route should be removed/stubbed to avoid dead functionality.

---

## 11. MEDIUM — `_create_chat_data_table` Never Substitutes `{context_table}` / `{embedding_dim}` Placeholders

**File:** `contract_finder_agent.py` (line 815), `queries/create_chat_data_table.sql`

The SQL template declares `{context_table}` and `{embedding_dim}` placeholders (lines 1, 6, 15, 17, 19 of the `.sql` file), but `_create_chat_data_table()` passes the raw, unformatted query string directly to `_execute_query` (`query = self.queries['create_chat_data_table']`, no `.format(...)` call) — this will fail at the database driver level with a syntax error (literal `{context_table}` is not valid SQL) every time the table-initialization step runs, which happens unconditionally at the top of every `run()` call (line 562).

---

## 12. MEDIUM — `_upsert_retrieved_documents` Does Not Use `upsert_retrieved_documents.sql`'s Expire Step Nor Match `insert_retrieved_documents.sql`'s Placeholder Contract

**File:** `contract_finder_agent.py` (lines 872-912), `queries/upsert_retrieved_documents.sql`, `queries/insert_retrieved_documents.sql`

ACT 10 explicitly requires a two-step expire-then-insert flow: first execute `upsert_retrieved_documents.sql` (which sets `is_expired = TRUE` for prior session rows) then `insert_retrieved_documents.sql` for the new batch. The implemented `_upsert_retrieved_documents()` **skips the expire step entirely** — `upsert_retrieved_documents.sql` is never loaded/executed anywhere in `contract_finder_agent.py` (confirmed via grep, only referenced by filename inside the queries directory itself). Additionally, `insert_retrieved_documents.sql`'s template expects a `{values_clause}` placeholder for a multi-row `VALUES (...)` clause with 9 positional columns (`session_id, doc_id, question, query_embeddings, doc_context, is_displayed, is_expired, created_date, context_score`), but the Python code instead calls `.format(session_id=..., question=..., documents_json=...)` — none of which match the SQL template's actual `{values_clause}` placeholder — meaning **this `.format()` call will raise `KeyError: 'values_clause'`** at runtime. This confirms the persistence layer (Step A6) is entirely broken independent of Findings #1-3.

---

## 13. LOW — Duplicate/Redundant Doc-String Comments Referencing Non-Existent Spec Sections

Several docstrings (e.g. `_retrieve_ids`, lines 200-203) reference "Per Spec Point 3/4/5" without those points being co-located or cross-referenced anywhere retrievable in the codebase — acceptable for traceability during generation, but should be cleaned up or moved to a dedicated ADR/decision-log document for production maintainability.

---

## Summary Table — ACT-by-ACT Business Logic Verification

| ACT | Title | Status |
|---|---|---|
| 1 | Runtime Environment Init | OK — `Config.py`, `requirements.txt`, `run.py` present and structurally sound |
| 2 | Route Registration & Payload Validation | Missing dedicated 400-on-missing-`user_query` validation |
| 3 | Agent Setup & Config Init | OK — tools initialized correctly in `setup()` |
| 4 | Filter Mapping & Query Construction | `_retrieve_ids`/`_build_filter_query` present but "associated" branch is dead code; metadata exact-match filters unused (Finding #8) |
| 5 | Semantic Search & Intersection | OK — intersection (AND) logic in `_retrieve_ids` implemented correctly |
| 6 | Content Chunk Retrieval | BROKEN — method exists only in orphaned files (Finding #1) AND its mapping key is missing (Finding #3) |
| 7 | Score Calculation & Ranking | BROKEN — same orphaned-file issue (Finding #1); duplicate divergent logic (Finding #2) |
| 8 | Context Assembly | BROKEN — same orphaned-file issue (Finding #1) |
| 9 | LLM Invocation & Response Handling | Timeout, sentinel-detection, and no-context-template logic all absent (Finding #4) |
| 10 | Session Document Persistence | Expire step skipped; insert `.format()` call will `KeyError` (Finding #12) |
| 11 | Citation Generation & Extraction | Simplified stub, no exact-match verification, no threading, no prompt file (Finding #5) |
| 12 | Pagination Branching & Flow Routing | Present but lacks the spec-mandated string/bool normalization for `show_more_details` |
| 13 | Session History Retrieval & Formatting | OK — `_fetch_show_more_documents` structurally reasonable |
| 14 | Pagination Response Assembly | OK — `_show_more_details` correctly separates displayed/new docs |
| 15 | Shared Response Assembly | OK — `run.py`'s `predict_api()` assembles response/documents/citations/metadata per spec shape |

---

## Priority Remediation Order

1. **Finding #1 + #2** — Move ACT 6/7/8 methods into `ContractFinderAgent` class body; delete orphaned files; reconcile duplicate logic. (Blocks 100% of Flow A requests.)
2. **Finding #3** — Add `contract_content_chunks` mapping entry to `FILTER_TABLE_MAPPING`. (Blocks all content retrieval even after fix #1.)
3. **Finding #6** — Add missing `smart_response_adjustment` and `max_contracts_to_return` fields to `ContractFinderAgentSetup`. (Blocks every request's return path.)
4. **Finding #12** — Fix `insert_retrieved_documents.sql` placeholder mismatch and wire in the expire step. (Blocks all session persistence.)
5. **Finding #4** — Implement LLM timeout wrapper + sentinel detection + no-context template substitution per ACT 9.
6. **Finding #5** — Rebuild `_generate_citations` per ACT 11's full spec (regex sentence extraction, threaded `CitationTool.run()` calls, missing-doc fallback, dedicated prompt file).
7. **Finding #7, #8, #9, #10, #11** — Secondary correctness/completeness gaps to close before considering this production-ready.

---

*Report generated via static code review comparing all 15 ACT specifications, the `Book_Finder_2` reference implementation, and the generated `agents/contract_finder_agent/` codebase. No runtime/integration testing was performed as part of this review — all findings are based on static call-graph and cross-reference analysis.*

