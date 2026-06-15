"""Future differential-evolution search implementation.

The MVP keeps search orchestration outside this module.  This file exists to make
that extension point explicit.  The intended implementation will operate on a
flat categorical encoding of Candidate params and use DB-backed benchmark
results as the objective.
"""
