"""Opt-in, instance-only post-smoothing seam. No FluidGym import or CFD at import."""
import numpy as np


def array(x):
    return x.detach().cpu().numpy().copy() if hasattr(x, 'detach') else np.asarray(x).copy()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class InstancePatch:
    """Restore the exact previous instance namespace, including inherited methods."""
    def __init__(self, obj, name, replacement):
        self.obj, self.name = obj, name
        self.present = name in vars(obj)
        self.previous = vars(obj).get(name)
        setattr(obj, name, replacement)

    def close(self):
        if self.present:
            setattr(self.obj, self.name, self.previous)
        else:
            delattr(self.obj, self.name)


def gain_at(arm, t):
    require(arm in ('N-hook', 'F', 'J'), 'unsupported gain schedule')
    require(type(t) is int and 0 <= t < 12, 'transition outside reviewed pilot')
    return .5 if arm == 'F' or (arm == 'J' and t >= 6) else 1.


class PostSmoothingAdapter:
    """Caller must pass pinned-source gate before attaching to a native instance.

    No command scaling: raw.step receives its original argument, and only the
    input to the original _apply_action is multiplied. Smoother memory is read only.
    """
    def __init__(self, raw, arm):
        gain_at(arm, 0)
        require(not getattr(raw, '_stage1_adapter_attached', False), 'adapter already attached')
        self.raw, self.arm = raw, arm
        self.phase, self.t, self.gain = 'uninitialized', 0, 1.
        self.arguments = []
        self.original_apply, self.original_reset, self.original_step = raw._apply_action, raw.reset, raw.step
        self.patches = [InstancePatch(raw, '_stage1_adapter_attached', True),
                        InstancePatch(raw, '_apply_action', self.apply),
                        InstancePatch(raw, 'reset', self.reset),
                        InstancePatch(raw, 'step', self.step)]

    def memory(self):
        return array(self.raw._CylinderEnvBase__last_control)

    def apply(self, control):
        require(self.phase in ('initializing', 'stepping'), 'actuation outside reset/transition')
        before = self.memory()
        effective = control * self.gain
        result = self.original_apply(effective)
        require(np.array_equal(before, self.memory()), 'adapter/native apply changed smoother memory')
        if self.phase == 'stepping':
            self.arguments.append({'smoothed_argument': array(control),
                                   'effective_argument_to_native_apply': array(effective),
                                   'gain': self.gain})
        return result

    def reset(self, *args, **kwargs):
        self.phase, self.t, self.gain, self.arguments = 'initializing', 0, 1., []
        try:
            result = self.original_reset(*args, **kwargs)
            require(np.array_equal(self.memory(), np.zeros(1, np.float32)), 'reset smoother memory not zero')
            for side in ('top', 'bottom'):
                boundary = getattr(self.raw, f'_{side}_boundary')
                require(np.all(array(boundary.velocity) == 0), 'reset boundary readback not zero')
            self.phase = 'armed'
            return result
        except BaseException:
            self.phase = 'failed'
            raise

    def step(self, action):
        require(self.phase == 'armed', 'adapter requires successful reset; no retry after failure')
        require(int(self.raw._n_steps) == self.t, 'adapter/native transition index mismatch')
        self.gain, self.arguments, self.phase = gain_at(self.arm, self.t), [], 'stepping'
        try:
            result = self.original_step(action)
            require(len(self.arguments) == 25, 'native actuation count differs from pinned contract')
            require(int(self.raw._n_steps) == self.t + 1, 'native transition counter mismatch')
            self.t += 1
            self.phase = 'armed'
            return result
        except BaseException:
            self.phase = 'failed'
            raise

    def close(self):
        for patch in reversed(self.patches):
            patch.close()
        self.patches.clear()
        self.phase = 'closed'
