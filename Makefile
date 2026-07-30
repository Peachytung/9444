SHELL := /bin/bash

include hpc/.env.hpc2

TAR ?= tar
SSH_OPTS ?= -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=120 -o TCPKeepAlive=yes
REMOTE_DATA_ZIP := $(HPC2_ROOT)/assets/UECFOODPIX.zip
REMOTE_DATA_ROOT := $(HPC2_ROOT)/assets/UECFOODPIX/data
REMOTE_ITEMS := README.md requirements.txt AGENTS.md src scripts hpc

.PHONY: hpc-ping hpc-bootstrap hpc-sync-code hpc-sync-data hpc-extract-data hpc-python-deps hpc-prepare-splits hpc-probe hpc-smoke hpc-benchmark hpc-train-debug hpc-train-debug-chain hpc-train hpc-status hpc-logs hpc-package hpc-fetch

hpc-ping:
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'hostname && whoami && pwd'

hpc-bootstrap:
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'mkdir -p "$(HPC2_ROOT)" "$(HPC2_ROOT)/assets" "$(HPC2_ROOT)/logs/slurm" "$(HPC2_ROOT)/outputs" "$(HPC2_ROOT)/tmp" "$(HPC2_ROOT)/results_packages"'

hpc-sync-code: hpc-bootstrap
	$(TAR) -cf - $(REMOTE_ITEMS) | $(SSH) $(SSH_OPTS) $(HPC2_HOST) 'tar --overwrite -xf - -C "$(HPC2_ROOT)"'

hpc-sync-data: hpc-bootstrap
	$(SCP) $(SSH_OPTS) "$(LOCAL_DATA_ZIP)" $(HPC2_HOST):"$(REMOTE_DATA_ZIP)"

hpc-extract-data: hpc-sync-code
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'PROJECT_ROOT="$(HPC2_ROOT)" bash "$(HPC2_ROOT)/hpc/ops/remote/extract_dataset.sh"'

hpc-python-deps: hpc-sync-code
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'PROJECT_ROOT="$(HPC2_ROOT)" PYTHON_BIN="$(HPC2_PYTHON)" bash "$(HPC2_ROOT)/hpc/ops/remote/setup_python_packages.sh"'

hpc-prepare-splits: hpc-sync-code
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'PROJECT_ROOT="$(HPC2_ROOT)" DATA_ROOT="$(REMOTE_DATA_ROOT)" PYTHON_BIN="$(HPC2_PYTHON)" bash "$(HPC2_ROOT)/hpc/ops/remote/prepare_splits.sh"'

hpc-probe: hpc-sync-code
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'PROJECT_ROOT="$(HPC2_ROOT)" PYTHON_BIN="$(HPC2_PYTHON)" PYTHONPATH="$(HPC2_ROOT)/python_packages" bash "$(HPC2_ROOT)/hpc/ops/remote/probe.sh"'

hpc-smoke: hpc-sync-code
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'cd "$(HPC2_ROOT)" && sbatch hpc/slurm/smoke.slurm'

hpc-benchmark: hpc-sync-code
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'cd "$(HPC2_ROOT)" && sbatch hpc/slurm/benchmark_epoch.slurm'

hpc-train-debug: hpc-sync-code
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'cd "$(HPC2_ROOT)" && sbatch hpc/slurm/train_baselines_debug.slurm'

hpc-train-debug-chain: hpc-sync-code
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'cd "$(HPC2_ROOT)" && bash hpc/ops/remote/submit_debug_chain.sh 8'

hpc-train: hpc-sync-code
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'cd "$(HPC2_ROOT)" && sbatch hpc/slurm/train_baselines.slurm'

hpc-status:
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'squeue -u $$(id -un); echo ===RECENT===; sacct -u $$(id -un) --starttime today --format=JobID,JobName,Partition,State,Elapsed,ExitCode -X | tail -n 20'

hpc-logs:
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'cd "$(HPC2_ROOT)" && find logs -maxdepth 2 -type f -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort | tail -n 20; echo ===LATEST_SLURM===; latest=$$(find logs -maxdepth 2 -type f -name "*.out" -printf "%T@ %p\n" | sort -n | tail -n 1 | cut -d" " -f2-); if [ -n "$$latest" ]; then tail -n 60 "$$latest"; fi; echo ===RESNET50===; if [ -f logs/resnet50_train.log ]; then tail -n 40 logs/resnet50_train.log; fi; echo ===VGG16===; if [ -f logs/vgg16_train.log ]; then tail -n 40 logs/vgg16_train.log; fi'

hpc-package: hpc-sync-code
	$(SSH) $(SSH_OPTS) $(HPC2_HOST) 'PROJECT_ROOT="$(HPC2_ROOT)" bash "$(HPC2_ROOT)/hpc/ops/remote/package_results.sh"'

hpc-fetch:
	mkdir -p incoming
	$(SCP) $(SSH_OPTS) $(HPC2_HOST):"$(HPC2_ROOT)/results_packages/comp9444_baseline_results_latest.tar.gz" incoming/
