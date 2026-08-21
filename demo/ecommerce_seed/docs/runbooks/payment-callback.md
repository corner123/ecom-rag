# Payment callback incident runbook

For repeated callbacks, search logs by PayGate `event_id` and confirm whether
the handler returned `duplicate`. Do not replay a callback by changing its
identifier. For amount mismatch, keep the order in `pending_payment`, record
the provider payment identifier, and escalate for reconciliation.

The synthetic demo has no production log system or automatic retry queue.
Operational metrics, customer impact, SLA, and recovery-time claims are out of
scope unless a dated experiment supplies them.
