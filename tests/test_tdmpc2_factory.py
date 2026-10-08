"""CPU factory regression using the pinned patched source and a non-forwarding wrapper.

Run without pytest: PYTHONPATH=src python -m unittest discover -s tests -p test_tdmpc2_factory.py -v
The fake environment is not evidence of CUDA simulator compatibility.
"""
import ast
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fluidgym_rl.upstream import make_tdmpc2_workcopy, repo_root, third_party, upstream_status


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class NonForwardingWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped


class FactoryAssertions:
    def check_factory(self, path, logger):
        tree = ast.parse(path.read_text())
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_env")
        # Execute the actual upstream factory; replace optional simulator factories only.
        def unsupported(cfg):
            raise ValueError(cfg.task)
        for horizon in (1, 5, 80):
            with self.subTest(horizon=horizon):
                inner = SimpleNamespace(max_episode_steps=horizon,
                                        observation_space=SimpleNamespace(shape=(6,)),
                                        action_space=SimpleNamespace(shape=(1,)))
                inner.unwrapped = inner
                namespace = {name.id: unsupported for name in ast.walk(factory)
                             if isinstance(name, ast.Name) and name.id.startswith("make_")}
                namespace.update(gym=SimpleNamespace(logger=logger), TensorWrapper=NonForwardingWrapper,
                                 make_fluidgym_env=lambda cfg: inner)
                exec(compile(ast.Module(body=[factory], type_ignores=[]), str(path), "exec"), namespace)
                cfg = Config(task="fluidgym-CylinderJet2D-easy-v0", multitask=False, obs="state", seed=0)
                env = namespace["make_env"](cfg)
                self.assertIsInstance(env, NonForwardingWrapper)
                self.assertIs(type(env.max_episode_steps), int)
                self.assertGreater(env.max_episode_steps, 0)
                self.assertEqual(env.max_episode_steps, horizon)
                self.assertEqual(cfg.episode_length, env.max_episode_steps)


@unittest.skipUnless((third_party() / "tdmpc2" / ".git").exists(), "pinned TD-MPC2 checkout unavailable")
class FactoryEpisodeLengthTests(FactoryAssertions, unittest.TestCase):
    def test_wrapped_factory_preserves_episode_length(self):
        before = upstream_status("tdmpc2")
        with tempfile.TemporaryDirectory() as directory:
            info = make_tdmpc2_workcopy(Path(directory) / "workcopy")
            self.assertEqual(info["changed_files"], ["tdmpc2/common/layers.py", "tdmpc2/envs/__init__.py"])
            path = Path(info["code_dir"]) / "envs" / "__init__.py"
            self.check_factory(path, SimpleNamespace())
        self.assertEqual(upstream_status("tdmpc2"), before)
        self.assertTrue(before["matches"])
        self.assertFalse(before["dirty"])


class SuppliedSourceTests(FactoryAssertions, unittest.TestCase):
    """Patch check against the user-supplied source, not a verified upstream clone."""
    def test_patch_and_both_logging_interfaces(self):
        source = repo_root() / "tests/fixtures/tdmpc2_envs_init.py.txt"
        patch = repo_root() / "patches/tdmpc2_fluidgym_task.patch"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tdmpc2/envs/__init__.py"
            path.parent.mkdir(parents=True)
            path.write_text(source.read_text())
            layers = root / "tdmpc2/common/layers.py"
            layers.parent.mkdir(parents=True)
            layers.write_text((repo_root() / "tests/fixtures/tdmpc2_ensemble.py.txt").read_text())
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
            subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
            modern = SimpleNamespace(min_level=0)
            self.check_factory(path, modern)
            self.assertEqual(modern.min_level, 40)
            levels = []
            legacy = SimpleNamespace(set_level=levels.append)
            self.check_factory(path, legacy)
            self.assertEqual(levels, [40, 40, 40])


if __name__ == "__main__":
    unittest.main()
