#!/usr/bin/env bash
# One-time setup for publishing this directory as a public repository.
# Create the empty repository on GitHub first (no README, no .gitignore),
# then run this from the repository root.
set -euo pipefail

USER="${1:-HAYDARKILIC}"
REPO="${2:-grokking}"

git init -b main
git add .
git commit -m "Code, figures, tables and result summaries for the nonlinear test of two weight-decay identities"
git remote add origin "https://github.com/${USER}/${REPO}.git"
git push -u origin main

echo
echo "Now, on GitHub:"
echo "  1. Settings > General > set the description and topics"
echo "  2. Releases > Draft a new release (tag v1.0.0) so the code has a fixed citable state"
echo "  3. Optionally connect Zenodo to mint a DOI, then put that DOI in the"
echo "     manuscript's Data and Code Availability statement and in CITATION.cff"
