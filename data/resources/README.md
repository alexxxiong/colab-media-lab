# Sanitized numeric resource samples

R01/R02 are anonymized research sessions. `elapsed_seconds` starts at the first record in each file; it is not a shared global clock across files. Worker CSVs measure the actual inference process tree. The R02 startup CSV follows the startup/kernel process tree and whole GPU during preparation and subsequent research.

Samples retain GPU utilization, board VRAM, power, temperature, and process-tree CPU/RSS/I/O. No wall-clock dates, PID lists, runtime endpoints or account identifiers are included. RSS can double-count shared pages. I/O counters describe live sampled processes and are not exact per-request deltas.

These measurements are research observations, not the unexecuted formal benchmark matrix.
