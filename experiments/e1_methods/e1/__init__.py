"""E1 public-benchmark ask-method runner (HumanChoice / medicaljiayi2).

Every metric produced by this package comes from executing a method on a task with the
configured OpenAI-compatible backend and scoring with the benchmark's own evaluator.
Nothing here models, interpolates or estimates a metric. Tasks that fail are stored with
status "error"/"timeout" and are excluded (and counted) by the aggregator, never imputed.
"""
__version__ = "1.0.0"
