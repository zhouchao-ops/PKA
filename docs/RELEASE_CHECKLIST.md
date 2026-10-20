# Open-source release checklist

## Must complete before publishing

- [x] Set the project name to PKA and repository to `zhouchao-ops/PKA`.
- [ ] Add a public contact email.
- [ ] Confirm copyright ownership and preserve upstream OminiControl notices.
- [ ] Review the FLUX.1-dev license and choose compatible licenses for the three LoRA adapters.
- [ ] Complete the training-data provenance and rights review in `dataset/README.md`.
- [ ] Add author names, paper title, arXiv/DOI, and citation metadata.
- [ ] Decide whether the dataset will be released, replaced with a preparation script, or withheld.
- [ ] Add a public evaluation protocol and results for canny+depth.
- [ ] Verify all public metrics on a clearly identified, redistribution-safe held-out set.
- [ ] Add representative example inputs/outputs after confirming image rights.

## Repository and model release

- [ ] Create a clean repository from this directory; do not publish the dirty research worktree or its history.
- [ ] Use Git LFS for `.safetensors` and media files.
- [ ] Replace placeholder project metadata in `pyproject.toml`.
- [ ] Add `CITATION.cff` and a security/contact policy.
- [ ] Run a secret scan and inspect the final Git history.
- [ ] Verify `sha256sum -c checkpoints/SHA256SUMS`.
- [ ] Build and install the wheel in a fresh environment.
- [ ] Run one known-good inference for each of the three checkpoints.
- [ ] Run unit tests and a GPU smoke test.
- [ ] Publish code, weights, and dataset as separate artifacts if their licenses differ.
- [ ] Create a version tag and immutable release archive.

## Recommended model-card content

- Intended uses and out-of-scope uses
- Base model and exact training configuration
- Training dataset, filtering, and preprocessing
- Evaluation datasets, metric implementations, and baselines
- Biases, limitations, safety risks, and environmental impact
- License, citation, contact, and version/checksum
