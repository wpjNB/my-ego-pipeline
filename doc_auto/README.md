# doc_auto

Auto-maintained project documentation. **Rule:** any code change must update the
relevant file here and append an explicit modification timestamp.

Last modified: 2026-10-11 20:47 (+08:00)

| File | Contents |
| --- | --- |
| [`architecture.md`](architecture.md) | Pipeline stages, coordinate conventions, on-disk contract |
| [`hot3d-gt-coordinate-frames.md`](hot3d-gt-coordinate-frames.md) | HOT3D MANO GT world-frame conversion and overlay refresh |
| [`environments.md`](environments.md) | Environment strategy (base / wilor / hawor / vggt) and device handling |
| [`setup.md`](setup.md) | Environment construction, checkouts, the weights download list, MANO |
| [`runbook.md`](runbook.md) | How to run both paths (CPU/mock and GPU/real), expected artefacts, troubleshooting |
| [`distributed.md`](distributed.md) | Batch dispatch: window sharding, content-addressed idempotency, hosts/executors, failure semantics |
| [`implementation-status.md`](implementation-status.md) | What is implemented, what is blocked and why |
| [`ablation.md`](ablation.md) | Mock-plumbing numbers and the table to fill with real data |
| [`changelog.md`](changelog.md) | Timestamped change log |
