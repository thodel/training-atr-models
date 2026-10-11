#!/bin/bash
# Forced command for the atr-results key (training-atr-models#214, A1).
#
# The results MCP on asteraix reaches this account as
#     ssh ubelix-direct python3 - <cmd> <json>
# with the probe source on stdin (src/atr_results_mcp/remote.py). That is the
# only thing this key may do. Anything else -- a shell, sbatch, scancel, scp,
# a second command after ";" -- is refused and logged.
#
# In ~/.ssh/authorized_keys:
#   command="/storage/homefs/th19c587/ubelix/atr_results_probe_only.sh",restrict ssh-ed25519 ... atr-results probe
#
# The check is a whitelist of characters, not a blacklist: after "python3 - "
# only letters, digits and the characters shlex.quote and JSON need. No "$",
# backtick, ";", "|", "&", "<", ">", "(", ")", "\" or newline can get through,
# so the eval below can expand nothing but the quotes.
export LC_ALL=C
LOG="$HOME/ubelix/atr_results_probe_only.log"
cmd="$SSH_ORIGINAL_COMMAND"
# Bracket expression: "]" must come first to be literal, "-" last; a backslash
# would be a literal character in there, not an escape.
re='^python3 - []A-Za-z0-9_'"'"'"{}:,. [/+=-]*$'
if [[ "$cmd" =~ $re ]]; then
    eval "exec $cmd"
fi
printf '%s refused: %q\n' "$(date -Is)" "$cmd" >> "$LOG"
echo "atr_results_probe_only: refused" >&2
exit 1
