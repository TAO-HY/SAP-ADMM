> Historical validation: the files in this directory were generated for version 1.0.1. In particular, general_a/ and original_package_general_a/ used the earlier 4001-update boundary. Version 1.0.2 changes that boundary to 4000; these archived outputs have not been recomputed. Current user-supplied table sources are in paper_reference/user_results/.

# Verification outputs

These files were computed on October 2, 2026, by the reviewed proposed-method implementation with paper dimensions and all default trials. They are separate from the supplied manuscript's Windows timing values and comparison figures. See each `metadata.json` for runtime details.

- Signal: 50 trials x 5 probabilities x 2 methods = 500 solves, plus illustrative recoveries.
- MNIST: 20 trials x 10 digits = 200 solves.
- General A: 20 trials x 5 Nmax settings = 100 solves.
- `results.npz` retains raw proposed-method outputs; CSV files retain full-precision metrics and diagnostics.
- The arrays use deterministic seeds. Times are environment-specific and include effects of concurrent verification jobs. They should not be used as a replacement benchmark for the original Windows timing entries.
- Comparison-method raw arrays are not available in the supplied archive.

To regenerate in a new directory, run `python -m experiments.run_all --mode compute --out results_new`. To plot these retained outputs, run `python -m experiments.run_all --mode plot --out verification`.
