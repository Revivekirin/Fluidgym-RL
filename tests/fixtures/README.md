`tdmpc2_envs_init.py.txt` is the TD-MPC2 environment factory supplied by the
user from their server checkout for this regression. Python indentation and
blank lines were restored from the pasted text and existing patch context.
It is a test fixture, not a byte-verified copy of the pinned Git blob.

The offline test applies the persistent patch to this fixture and executes
its factory with a non-forwarding wrapper and fake environment. The separate
pinned-checkout test invokes `make_tdmpc2_workcopy()` without mocking its SHA
or cleanliness checks; it skips when the real checkout is absent.
