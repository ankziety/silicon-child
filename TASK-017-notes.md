Task: TASK-017
Component: ai_infant/crawl/browser.py

Assumptions
- This change adds audit logging for browser actions by writing JobV1-like
  entries into the provided `Store` via `store.store_job(...)`.

What I changed
- `execute_action` now records low-confidence rejections and execution
  outcomes to the `Store` using the existing `_log_job` helper. Action
  attempts are still recorded in `action_history` for in-memory tracing.

Why
- Provide auditable server-side logs for action decisions (reject/execute)
  to aid debugging, monitoring and compliance.

Blockers / escalation
- None. This is a localized, low-risk change. It relies on the provided
  `Store` object implementing `store_job(job_data)` as used elsewhere in
  the repo.

Cost & alignment
- Low cost. No external network calls introduced. Logging volume is small
  (one job per attempted action).

Signed-off-by: Frontier engineer


