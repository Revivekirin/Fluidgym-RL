"""CPU regression checks for the preparation boundary, not shift experiments."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fluidgym_rl.environment_contract import CONTRACT, resolve_environment, validate_physics
from fluidgym_rl.tdmpc.environment import CONTRACT as LEGACY_CONTRACT, make_env
from fluidgym_rl.tdmpc.runtime import DEFAULT_CONFIG, namespace, read_config, repo_root


class EnvironmentContractTests(unittest.TestCase):
    def test_nominal_config_and_manifest_contract_are_unchanged(self):
        # Captured before extraction, includes all parameter provenance strings.
        digest = lambda data: hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest(read_config()), '928971c11935776a51cde532ed2d0291f490b5060017675e5edd34f5bde1f82f')
        self.assertEqual(digest(CONTRACT), '374648ce9fa92d29c1ece15d919fc99c589b9b8387a0014d3616757b7198691a')
        self.assertIs(CONTRACT, LEGACY_CONTRACT)

    def test_independent_spec_composes_without_algorithm_fields(self):
        path = repo_root() / 'configs/environments/cylinderjet2d_nominal.json'
        original = read_config(seed=7, device='cpu')
        composed = read_config(seed=7, device='cpu', environment_config=path)
        self.assertEqual(original, composed)
        self.assertEqual(resolve_environment(path), resolve_environment())
        self.assertEqual(vars(namespace(composed)), original['parameters'])
        self.assertNotIn('physics', vars(namespace(composed)))
        spec = resolve_environment()
        spec['physics']['reynolds_number'] = 250.0
        self.assertEqual(resolve_environment()['physics']['reynolds_number'], 100.0)

    def test_unsupported_or_malformed_spec_rejected(self):
        nominal = resolve_environment()
        changes = [('physics', 'reynolds_number', 250.0), ('physics', 'viscosity', .01),
                   ('physics', 'lift_penalty', float('nan')), ('physics', 'action_smoothing_alpha', .2),
                   ('interface', 'episode_length', 160), ('interface', 'flatten_observation', 1),
                   ('interface', 'observation_shape', [151, 2])]
        for section, key, value in changes:
            with self.subTest(key=key):
                spec = copy.deepcopy(nominal); spec[section][key] = value
                with self.assertRaisesRegex(ValueError, 'unsupported environment'):
                    resolve_environment(spec)
        for spec in ({}, [], {'schema_version': 2}):
            with self.assertRaises(ValueError): resolve_environment(spec)

    def test_misplaced_physics_rejected_in_algorithm_config(self):
        for where in ('parameters', 'root'):
            data = json.loads(DEFAULT_CONFIG.read_text())
            if where == 'parameters':
                data['parameters']['reynolds_number'] = 250
                data['provenance']['reynolds_number'] = {'source': 'unsupported'}
            else:
                data['physics'] = {'reynolds_number': 250}
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'config.json'; path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'unsupported config keys'):
                    read_config(path)

    def test_saved_resolved_configuration_remains_accepted(self):
        data = read_config()
        data['tracking'] = {'mode': 'disabled'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'resolved_config.yaml'; path.write_text(json.dumps(data))
            self.assertEqual(read_config(path), data)

    def test_physics_checks_preserve_tolerance_and_reject_missing_fields(self):
        values = dict(step_length=.25, _reynolds_number=100., _lift_penalty=1., _action_smoothing_alpha=.1)
        validate_physics(SimpleNamespace(**values))
        validate_physics(SimpleNamespace(**{**values, '_reynolds_number': 100.00001}))
        for key in values:
            for value in (float('nan'), float('inf'), values[key] * 2):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    validate_physics(SimpleNamespace(**{**values, key: value}))
            missing = dict(values); del missing[key]
            with self.assertRaisesRegex(ValueError, key): validate_physics(SimpleNamespace(**missing))

    def test_factory_rejects_overrides_before_simulator_creation(self):
        cfg = namespace(read_config(device='cpu'))
        with patch('fluidgym_rl.envs.make_fluid_env') as factory:
            for kwargs in ({'reynolds_number': 250}, {'viscosity': .01}, {'flatten': False}):
                with self.assertRaisesRegex(ValueError, 'unsupported TD-MPC environment'):
                    make_env(cfg, **kwargs)
            factory.assert_not_called()

    def test_cli_explicit_nominal_is_identical_and_shift_fails_before_run(self):
        cli = [sys.executable, str(repo_root() / 'scripts/tdmpc_train.py'), '--print-config']
        before = subprocess.run(cli, capture_output=True, text=True, check=True)
        explicit = subprocess.run(cli + ['--environment-config', str(repo_root() / 'configs/environments/cylinderjet2d_nominal.json')], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(before.stdout), json.loads(explicit.stdout))
        with tempfile.TemporaryDirectory() as tmp:
            spec = resolve_environment(); spec['physics']['reynolds_number'] = 250.0
            path = Path(tmp) / 'shift.json'; path.write_text(json.dumps(spec))
            failed = subprocess.run(cli + ['--environment-config', str(path)], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('only nominal is executable', failed.stderr)
            self.assertNotIn('Run directory:', failed.stdout)


if __name__ == '__main__':
    unittest.main()
