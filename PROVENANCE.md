# Provenance

The N12 workspace supplied the frozen nominal, three gate records, gate source, focused gate tests, and runtime modules used for this package.

`artifacts/n12-release-source-sha256.txt` lists the seven packaged source files retained from that workspace. `n12-release` requires that exact path set and recomputes every digest. `artifacts/MANIFEST.json` also records those source digests and the ledger digest.

`n12-verify` separately pins the five runtime files that execute the unperturbed rollout. It checks those digests and the frozen nominal before simulation.

Eight State publishes this package under the MIT license with a 2026 Alex Garcia Gil copyright notice.
