# Prescription image input — Priority #3

Camera and gallery/file images accept JPEG, PNG and WebP with consistent MIME
types, at most 10 MB and 24 megapixels. Every HTTP image endpoint uses bounded
decoding. Invalid files return 400, unsupported/mismatched MIME returns 415, and
size/dimension limits return 413 before OpenCV allocation for oversized headers.

- `POST /capture-image`: multipart `file`, `session_id` and optional camera
  metadata. Stores the original and registers a capture attempt. Quality is
  checked separately by `/validate-capture`, preserving the camera workflow.
- `POST /validate-capture`: multipart `file` and optional registered `image_id`.
  Returns quality feedback; metrics are recorded when an image ID is supplied.
- `POST /validate-upload`: multipart `file`. Checks a selected/cropped image
  without storing it, registering an attempt or recording database metrics.
- `POST /upload-file`: multipart `file`, `session_id`. Stores/registers an attempt
  and checks quality. Returns the image ID on success; rejected quality returns
  422 with the image ID, attempt number, reason and feedback.
- `GET /files/{image_id}`: retrieves the registered original from configured
  storage. Missing records/files return 404; paths outside storage are refused.

## One acceptance policy

`_validate_image_quality` applies the same lighting, blur, readability and
handwriting thresholds to camera and uploaded images. Framing is reported as
aligned, too small or not found; it is advisory for both sources. Uploads do not
have stricter framing requirements. The usable-image decision and rejection
reason are identical for identical decoded pixels. Live preview retains its
existing feedback contract; real-time document-gated capture remains future work.

## Storage and retention limits

`FILE_STORAGE_DIR` defaults to `CleaRx-backend/storage` beside the application.
Use an absolute persistent-volume path when configuring it. The repository's
default storage directory is ignored by Git. Custom storage paths need their own
access controls and exclusions. The database stores the path and image checksum
in existing prescription-image fields; no additional SQL migration is required.

Successful registration stores camera images and uploads, including quality-
rejected upload attempts. A failed registration removes the newly written file.
Files and records are not automatically deleted on session close or after a TTL.
Disk and database writes are not one atomic transaction. No masking, anonymization
or provider-side retention guarantee is implemented by this storage change.
Do not infer any of those guarantees from the app's workflow.

The existing backend has no authenticated ownership check for file retrieval.
Use only a trusted test environment until deployment authentication and retention
requirements are addressed. B8 adds the planned local/test runtime guard.
The external-processing notice is part of the later frontend migration.

## Verification

Tests compare camera and upload quality results for controlled conditions and
generated images using actual image-processing checks. They also cover file
limits, stored originals, registration cleanup, crop-preview side effects and
retrieval containment. No live database, patient images or OCR provider is used;
these tests do not establish improved capture or recognition accuracy.
