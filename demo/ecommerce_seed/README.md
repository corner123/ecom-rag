# Synthetic Commerce Engineering Demo

This repository is a synthetic, non-production fixture for demonstrating an
engineering-knowledge RAG. It contains no company data, customer data,
credentials, traffic measurements, or production claims.

The fictional platform has four bounded modules:

- order lifecycle and payment confirmation;
- inventory reservation, confirmation, and release;
- deterministic coupon calculation;
- signed, idempotent payment callbacks.

Documents describe design intent. Python modules remain the authority for the
current implementation, so the RAG can demonstrate stale-document detection
and live source verification.
