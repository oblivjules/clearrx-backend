# Recognition options — Priority #2

The API offers **CleaRx TrOCR** as its primary recognition option and
**Claude Sonnet 5.5** as its external option. Both may process prescription images
through a third-party AI service. The primary name is a product label, not a
guarantee of offline processing or evidence that a local TrOCR model executed.

`POST /process-image` accepts the existing multipart image/registration fields
and an optional `model` field:

| Value | Public name |
| --- | --- |
| `clearx-trocr` (default) | CleaRx TrOCR |
| `claude-sonnet-5-5` | Claude Sonnet 5.5 |

Older clients' primary option is accepted for migration compatibility. Unsupported
options return 422 before recognition. If the selected service is unavailable,
the backend attempts genuine local TrOCR and reports **Local TrOCR** when used.
Failure of both paths returns a no-text rejection with recovery guidance.
The same deterministic parser, catalog matcher and review policy process every
source. No switch to a local-first confidence-gated architecture is introduced.

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
