"""Real Torch/TensorDict regression, CPU by default; no simulator or training run."""
import importlib.util
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from fluidgym_rl.upstream import repo_root


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('tensordict'),
                     'Torch and TensorDict required')
class EnsembleMoveTests(unittest.TestCase):
    def test_weights_gradients_templates_and_targets(self):
        import torch
        from tensordict.nn import TensorDictParams
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target, fixture in [('tdmpc2/common/layers.py', 'tdmpc2_ensemble.py.txt'),
                                    ('tdmpc2/envs/__init__.py', 'tdmpc2_envs_init.py.txt')]:
                p = root / target
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text((repo_root() / 'tests/fixtures' / fixture).read_text())
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', 'apply', str(repo_root() / 'patches/tdmpc2_fluidgym_task.patch')], cwd=root, check=True)
            scope = {}
            exec(compile((root / 'tdmpc2/common/layers.py').read_text(), 'layers.py', 'exec'), scope)
            torch.manual_seed(7)
            modules = [torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.1), torch.nn.Linear(4, 2)) for _ in range(2)]
            x = torch.randn(5, 3)
            for m in modules: m.eval()
            expected = torch.stack([m(x) for m in modules])
            ensemble = scope['Ensemble'](modules).eval()
            before = [p.detach().clone() for p in ensemble.params.parameters()]
            # Mirrors WorldModel.init's detached and target ensemble construction.
            detached = TensorDictParams(ensemble.params.data, no_convert=True)
            target_params = TensorDictParams(ensemble.params.data.clone(), no_convert=True)
            with detached.data.to('meta').to_module(ensemble.module):
                target = deepcopy(ensemble)
            delattr(target, 'params')
            target.__dict__['params'] = target_params
            parent = torch.nn.Module()
            parent.online = ensemble
            parent.target = target
            parent.target_params = target_params
            parent.to('cpu')  # Previously raises Cannot copy out of meta tensor.
            for a, b in zip(before, ensemble.params.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(ensemble(x), expected)
            torch.testing.assert_close(target(x), expected)
            output = ensemble(x)
            self.assertTrue(output.requires_grad, "functional ensemble output lost its autograd connection")
            output.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in ensemble.params.parameters()))
            # Check actual gradient values, not just their existence or finiteness.
            expected.sum().backward()
            for index, module in enumerate(modules):
                for name, parameter in module.named_parameters():
                    stacked = ensemble.params[tuple(name.split('.'))]
                    torch.testing.assert_close(stacked.grad[index], parameter.grad)
            # Detached/target parameters must still allow gradients with respect
            # to inputs (policy update), without gradients into target weights.
            target_input = x.detach().clone().requires_grad_(True)
            target(target_input).sum().backward()
            self.assertTrue(torch.isfinite(target_input.grad).all())
            self.assertTrue(all(not p.requires_grad for p in target_params.values(True, True)))
            self.assertTrue(all(p.is_meta for p in ensemble.module.parameters()))
            ensemble.train()
            self.assertTrue(ensemble.module.training)
            ensemble.eval()
            self.assertFalse(ensemble.module.training)
            if torch.cuda.is_available():
                parent.to('cuda')
                torch.testing.assert_close(ensemble(x.cuda()).cpu(), expected)
                self.assertTrue(all(p.is_meta for p in ensemble.module.parameters()))
            # The finally clause must restore registration even when conversion fails.
            def fail(tensor):
                raise RuntimeError('conversion failed')
            with self.assertRaisesRegex(RuntimeError, 'conversion failed'):
                ensemble._apply(fail)
            self.assertIs(ensemble._modules['module'], ensemble.module)


if __name__ == '__main__':
    unittest.main()
