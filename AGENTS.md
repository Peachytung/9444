# HPC control-plane rules

- Treat this folder as the local control plane and HPC2 as the execution plane.
- Use the Makefile and `hpc/ops/remote/*.sh` for routine remote operations.
- Keep every remote artifact under the configured `HPC2_ROOT`.
- Never delete or overwrite unrelated remote paths.
- Run the smoke job before the full training job.
- Keep data, logs, checkpoints, outputs, and temporary files in separate subdirectories.
- Fetch results as one packaged archive instead of opening many SSH/SCP connections.

