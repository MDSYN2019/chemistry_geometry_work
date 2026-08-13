.PHONY: inspect test ligands edo-reference endpoints analyse-endpoints solvent-rbfe complex-rbfe analyse-rbfe
inspect:
	python structure/inspect_pdb.py data/raw/start2_refmac1.pdb

test:
	pytest

ligands edo-reference endpoints analyse-endpoints solvent-rbfe complex-rbfe analyse-rbfe:
	@echo "Target '$@' is intentionally gated: M0 is the current milestone." >&2
	@exit 2
