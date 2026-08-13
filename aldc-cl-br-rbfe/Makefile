.PHONY: inspect test openmm-dry-run ligands edo-reference endpoints analyse-endpoints solvent-rbfe complex-rbfe analyse-rbfe
inspect:
	python structure/inspect_pdb.py data/raw/start2_refmac1.pdb

test:
	pytest

openmm-dry-run:
	@test -n "$(PRMTOP)" -a -n "$(INPCRD)" || \
	  (echo "usage: make openmm-dry-run PRMTOP=... INPCRD=... [OUTPUT=...]" >&2; exit 2)
	python md/run_openmm.py --prmtop "$(PRMTOP)" --inpcrd "$(INPCRD)" \
	  --output-dir "$(or $(OUTPUT),md/edo_reference)" --dry-run

ligands edo-reference endpoints analyse-endpoints solvent-rbfe complex-rbfe analyse-rbfe:
	@echo "Target '$@' is intentionally gated: M0 is the current milestone." >&2
	@exit 2
