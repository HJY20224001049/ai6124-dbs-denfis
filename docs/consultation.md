# Project consultation — preparation

Status: pending the actual consultation on 2026-11-11. No instructor response
has been received or inferred. LIN Jingzhao should record the answers below.

1. Is predict-then-update evaluation using realised labels accepted as an online
   extension, with frozen DENFIS in the main comparison?
2. Does a self-written ECM/TSK/RLS implementation plus bounded rules and explicit
   merge/online experiments satisfy the course's established-method extension
   criterion? We do not claim to invent DENFIS.
3. Are long/flat trading and 10 bp one-way costs appropriate? How should forecast
   quality and risk-adjusted trading outcomes be balanced in the presentation?

Bring docs/results_table.md and docs/denfis_notes.md. Architecture:
13 causal features -> training-only preprocessing -> ECM -> Gaussian rule
weights -> first-order TSK -> return forecast; RLS estimates local consequents.

Answers: pending. Protocol changes: none authorised by the instructor yet.
