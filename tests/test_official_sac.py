import zipfile
import gymnasium as gym
import numpy as np
import pytest

from fluidgym_rl.official_sac import (CheckpointContractError, PUBLISHED, SB3Policy, TRAIN_SEEDS, inspect_zip, local_checkpoint, official_repo_id,
                                      resolve_official_checkpoint, sha256_file, validate_contract)


class M:  # minimal stand-in for a loaded SB3 model
    def __init__(self, obs, act): self.observation_space, self.action_space = obs, act
    def predict(self, o, deterministic=True): self.last = (o.dtype, deterministic); return np.array([0.1], np.float32), None


def box(n): return gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
def act(lo=-1, hi=1): return gym.spaces.Box(lo, hi, (1,), np.float32)
class E:  # env stand-in
    def __init__(self, n=302): self.observation_space, self.action_space = box(n), act()


def test_matching_contract_passes_and_marks_order_unverified():
    notes = validate_contract(M(box(302), act()), E(), ckpt_files=["0/ckpt_latest.zip"], installed_sb3="2.7.0", saved_sb3="2.7.0")
    assert "UNVERIFIED" in notes["observation_field_order"]


def test_observation_dim_mismatch_diagnostic():
    with pytest.raises(CheckpointContractError, match=r"\(300,\).*\(302,\)"): validate_contract(M(box(300), act()), E())


def test_structured_dict_observation_is_refused_not_flattened():
    dict_space = gym.spaces.Dict({"velocity": box(2)})
    with pytest.raises(CheckpointContractError, match="refusing to guess"): validate_contract(M(dict_space, act()), E())


def test_action_bounds_and_normalization_artifacts_and_sb3_major_are_reported_together():
    with pytest.raises(CheckpointContractError) as ei:
        validate_contract(M(box(302), act(-2, 2)), E(), ckpt_files=["0/vecnormalize.pkl"], installed_sb3="3.0.0", saved_sb3="2.7.0")
    msg = str(ei.value)
    assert "action bounds" in msg and "vecnormalize" in msg and "SB3 major" in msg


def test_missing_local_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError): local_checkpoint(tmp_path / "nope.zip")


def test_inspect_zip_reads_metadata_without_unpickling(tmp_path):
    p = tmp_path / "c.zip"
    with zipfile.ZipFile(p, "w") as z: z.writestr("_stable_baselines3_version", "2.7.0\n"); z.writestr("data", "{}")
    assert inspect_zip(p)["sb3_version_saved"] == "2.7.0" and len(sha256_file(p)) == 64
    assert local_checkpoint(p, 3)["train_seed"] == 3


def test_invalid_seed_and_repo_naming():
    with pytest.raises(ValueError): resolve_official_checkpoint("CylinderJet2D-easy-v0", 7)
    assert official_repo_id("CylinderJet2D-easy-v0") == "safe-autonomous-systems/sac-CylinderJet2D-easy-v0" and TRAIN_SEEDS == (0, 1, 2, 3, 4)


def test_sb3_policy_modes_and_dtype():
    m = M(box(3), act()); p = SB3Policy(m, deterministic=True)
    assert p.mode == "deterministic" and p.act(np.zeros(3, np.float64), True).shape == (1,) and m.last == (np.float32, True)
    assert SB3Policy(m, deterministic=False).mode == "stochastic"


def test_published_reference_transcription_sanity():
    pc = PUBLISHED["model_card_per_seed"]; assert [pc[s]["mean_reward"] for s in range(5)] == [0.05, 0.05, 0.05, 0.04, 0.05]
    t8 = PUBLISHED["paper_table8_iqm"]; assert (t8["mean_reward"], t8["drag"], t8["lift"], t8["baseflow_drag"]) == (0.051, 3.105, 0.032, 3.328)
