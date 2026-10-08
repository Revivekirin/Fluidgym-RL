`tdmpc2_envs_init.py.txt` is the TD-MPC2 environment factory supplied by the
user from their server checkout for this regression. Python indentation and
blank lines were restored from the pasted text and existing patch context.
It is a test fixture, not a byte-verified copy of the pinned Git blob.

The offline test applies the persistent patch to this fixture and executes
its factory with a non-forwarding wrapper and fake environment. The separate
pinned-checkout test invokes `make_tdmpc2_workcopy()` without mocking its SHA
or cleanliness checks; it skips when the real checkout is absent.

`tdmpc2_ensemble.py.txt` contains the supplied `Ensemble` class and imports
from `common/layers.py`, for testing the meta-template movement fix. It is
a source excerpt, not a full upstream file.

`tdmpc2_logging_source.py.txt` is the user-supplied logger initialization excerpt;
`tdmpc2_pilot_config.yaml` transcribes the relevant keys from the supplied Hydra
config. They support offline patch/override regression, not verification of a
complete upstream checkout. `--validate-only` composes the real pinned config.

The logger fixture now preserves the CRLF endings confirmed from the server
(241/241 upstream lines). Use read_bytes/write_bytes when copying it; text I/O
would hide the patch incompatibility this regression is intended to detect.
