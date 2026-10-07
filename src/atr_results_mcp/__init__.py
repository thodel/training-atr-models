"""A reading MCP for training results on UBELIX (#156, #174).

Two halves, deliberately separate:

- :mod:`atr_results_mcp.probe` runs **on the cluster's login node** and is written
  for its Python 3.9 and nothing but the standard library. It answers one question
  per invocation and prints JSON. It never submits, cancels or deletes anything.
- :mod:`atr_results_mcp.server` runs wherever the MCP client is (the laptop today,
  asteraix behind tei later) and turns each question into a tool. It ships the
  probe's source over ``ssh … python3 -`` on every call, so the cluster needs no
  checkout of this package and the two halves cannot drift apart.

Why an MCP and not ``ssh`` in a prompt: an unattended session can only call what
its allow rules name. A Bash rule names a command *line*; one changed flag and the
run waits for a click nobody gives (#174 counts the days this cost). An MCP tool
is allowed once, by name, whatever its arguments.
"""
