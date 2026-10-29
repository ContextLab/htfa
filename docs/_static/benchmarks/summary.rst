.. list-table:: Full-run TFA fits, three adults and two repetitions per mode
   :header-rows: 1

   * - K
     - Exact: mean (range), seconds
     - Fast: mean (range), seconds
     - Ratio of total exact / fast time
   * - 50
     - 24.16 (16.67–29.38)
     - 7.35 (5.97–10.35)
     - 3.29
   * - 100
     - 135.27 (46.38–223.72)
     - 13.46 (10.73–15.30)
     - 10.05

The runner has 4 x86-64 virtual CPUs and uses Linux, Python 3.12.15,
NumPy 2.5.4, SciPy 1.18.1, and OpenBLAS (two threads). All 24 fits pass
the interference checks and stop on the parameter threshold. Ranges include
both participant differences and repeated timings; they are not confidence
intervals. Paired exact/fast ratios range from 2.72 to 4.28
at K = 50 and from 4.30 to 14.93 at K = 100.

The relative objective difference ``(fast - exact) / abs(exact)`` ranges from
-0.000210 to +0.000240 at K = 50 and from -0.000010 to +0.000213 at K = 100.
Fast mode therefore reaches both lower and higher objectives in these fits.
This timing comparison does not measure held-out accuracy.

Download the :download:`per-fit CSV </_static/benchmarks/cpu.csv>` or the
:download:`settings and provenance JSON </_static/benchmarks/cpu.json>`.
The :download:`raw CPU monitoring samples </_static/benchmarks/cpu-monitoring.json>`
allow the interference checks to be inspected independently.
