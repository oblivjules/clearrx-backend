# Recognition options — Priority #2

The API offers **CleaRx TrOCR** as genuine local recognition on the backend and
**Claude Sonnet 5.5** as the explicitly selected external option. This changes
the previously provider-backed primary option following the user's approval.
It is not an automatic confidence-gated hybrid.

`POST /process-image` accepts existing multipart image/registration fields and
an optional `model` field:

| Value | Public name | Engine |
| --- | --- | --- |
| `clearx-trocr` (default) | CleaRx TrOCR | Local TrOCR |
| `claude-sonnet-5-5` | Claude Sonnet 5.5 | External vision service |

Only these public identifiers are accepted. Removed provider aliases return 422;
clients using the documented primary identifier remain compatible. Local primary
failure returns an unavailable/no-text result without sending the image to a
provider. If the explicitly selected external option fails, the existing genuine
local fallback is retained and reported as **Local TrOCR**. All sources use the
same deterministic parser, catalog matcher and review policy.

## Local engine configuration

`TROCR_BASE_MODEL` defaults explicitly to `microsoft/trocr-base-handwritten`.
`TROCR_CHECKPOINT_DIR` may select a local TrOCR-compatible checkpoint directory;
a missing explicit directory is an error, not a silent model substitution.
Without this override, existing `runs/trocr-rx/best` or
`runs/trocr-rx/checkpoint-epoch-20` checkpoints are detected in that order;
otherwise the configured base model is used. Existing line-level base-model
fallback inside the local pipeline remains unchanged. Registry metadata records
configured checkpoint/base paths and does not invent training-dataset attribution.

Model artifacts/dependencies must be available to the backend before use. First
use may download base-model artifacts; provision/cache them before deployment.
No actual model download or inference was performed by the mocked regressions.
The external option still requires server-side `ANTHROPIC_API_KEY` credentials.
Its configured provider identifier is `claude-sonnet-5-5`, as listed in the
[provider documentation](https://platform.claude.com/docs/en/models/sonnet-5-5/overview).
Account access and live inference remain unverified.

## Confidence and provenance

`ocr_source` reports the option that actually supplied the reading, or
`Unavailable` when neither path produced one. Provider results have
`ocr_confidence=null` and `ocr_confidence_basis=unavailable`. The legacy
`trocr_confidence=0` is a compatibility placeholder in that case, not a measured
zero-confidence result. Local TrOCR scores use `uncalibrated_heuristic`.
Name similarity remains separate from recognition confidence and review flags.

The legacy database requires numeric confidence and currently stores a neutral
0.5 placeholder when provider confidence is unavailable. It is not a probability.
Internal model records retain technical attribution separately from public names.
Nullable confidence/provenance schema work remains outstanding.

## Privacy and operational limits

Keep provider credentials on the backend. The provider connection uses HTTPS;
this does not imply the app-to-backend connection is encrypted in local testing.
Images may contain personal information. This migration adds neither masking nor
automatic deletion and makes no provider-side retention or training guarantee.
See IMAGE_UPLOAD.md for actual storage behavior. The app must explain external
processing before submission; the notice is part of the frontend migration.

Provider and local-model availability, latency and clinical reading accuracy
require end-to-end verification. Migration tests mock providers and persistence;
they do not establish improved accuracy or calibrated confidence.
