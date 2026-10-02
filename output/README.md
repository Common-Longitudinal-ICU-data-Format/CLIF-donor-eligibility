## Output directory

Results are split into two folders by **data-sharing safety** — the names encode whether the contents
may leave your site:

* **`../<site>_upload_to_box/`** — **aggregate, shareable** results only, at the
  repository root rather than in here so it is obvious what to drag to Box.
  Everything delivered to the project PI / consortium goes there: no `patient_id`
  or row-level records. Start with its `provenance.md`.

* **[`intermediate_phi/`](intermediate_phi)** — **patient-level working data (NEVER share).** Filtered
  cohorts and any per-patient tables live here. Its contents are **git-ignored** so they can't be
  committed, and they must never be uploaded to Box or sent to the PI. See
  [`intermediate_phi/README.md`](intermediate_phi/README.md).

See [`../guides/primer.md`](../guides/primer.md) for the full data-security rules.
