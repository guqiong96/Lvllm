# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTeDSL compile-arch pinning (heterogeneous-TP arch anchoring).

All CPU-runnable: CuTeDSL resolves its compile arch from visible device 0, so on
a mixed-arch TP group the sm120 ranks used to compile sm_86 PTX and ptxas
rejected `cvt.e4m3x2.f32`. Only the pure-python decision is covered here, not
the DSL's codegen.
"""

import os

import pytest

import vllm.utils.cutedsl_arch as ca


class _Envar:
    def __init__(self, arch):
        self._arch = arch

    @property
    def arch(self):
        return self._arch

    @arch.setter
    def arch(self, value):
        self._arch = value


class _Dsl:
    def __init__(self, arch):
        self.envar = _Envar(arch)


def _raise_no_device(_index=None):
    raise RuntimeError("no device")


def _patch_device(monkeypatch, capability):
    if capability is None:
        monkeypatch.setattr(
            ca.torch.cuda,
            "get_device_capability",
            _raise_no_device,
        )
    else:
        monkeypatch.setattr(
            ca.torch.cuda,
            "get_device_capability",
            lambda _index=None: capability,
        )
    monkeypatch.setattr(ca.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ca.torch.cuda, "current_device", lambda: 0)


@pytest.mark.parametrize(
    ("capability", "expect"),
    [
        ((8, 0), "sm_80"),
        ((8, 6), "sm_86"),
        ((8, 9), "sm_89"),
        ((9, 0), "sm_90a"),
        ((10, 0), "sm_100a"),
        ((12, 0), "sm_120a"),
    ],
)
def test_cutedsl_arch_naming(monkeypatch, capability, expect):
    _patch_device(monkeypatch, capability)
    assert ca.cutedsl_arch() == expect


def test_cutedsl_arch_unknown_device(monkeypatch):
    _patch_device(monkeypatch, None)
    assert ca.cutedsl_arch() is None


def test_pin_fixes_dsl_that_answered_for_device_zero(monkeypatch):
    monkeypatch.delenv(ca._CUTEDSL_ARCH_ENV, raising=False)
    monkeypatch.setattr(ca, "_ENV_AT_IMPORT", None)
    _patch_device(monkeypatch, (12, 0))
    dsl = _Dsl("sm_86")  # what detection answers on a 3090 + 5060 Ti box
    monkeypatch.setattr(ca, "_built_dsls", lambda: [dsl])

    assert ca.pin_cutedsl_arch() == "sm_120a"
    assert dsl.envar.arch == "sm_120a"
    import os

    assert os.environ[ca._CUTEDSL_ARCH_ENV] == "sm_120a"


def test_pin_leaves_matching_dsl_alone(monkeypatch):
    monkeypatch.delenv(ca._CUTEDSL_ARCH_ENV, raising=False)
    monkeypatch.setattr(ca, "_ENV_AT_IMPORT", None)
    _patch_device(monkeypatch, (8, 6))
    dsl = _Dsl("sm_86")
    monkeypatch.setattr(ca, "_built_dsls", lambda: [dsl])

    assert ca.pin_cutedsl_arch() == "sm_86"
    assert dsl.envar.arch == "sm_86"


def test_explicit_env_wins_over_device(monkeypatch):
    monkeypatch.setenv(ca._CUTEDSL_ARCH_ENV, "sm_100a")
    monkeypatch.setattr(ca, "_ENV_AT_IMPORT", "sm_100a")
    _patch_device(monkeypatch, (12, 0))
    dsl = _Dsl("sm_120a")
    monkeypatch.setattr(ca, "_built_dsls", lambda: [dsl])

    assert ca.pin_cutedsl_arch() == "sm_100a"
    assert dsl.envar.arch == "sm_100a"


def test_pin_is_quiet_noop_without_cuda(monkeypatch):
    monkeypatch.delenv(ca._CUTEDSL_ARCH_ENV, raising=False)
    monkeypatch.setattr(ca, "_ENV_AT_IMPORT", None)
    monkeypatch.setattr(ca.torch.cuda, "is_available", lambda: False)
    dsl = _Dsl("sm_100a")
    monkeypatch.setattr(ca, "_built_dsls", lambda: [dsl])

    assert ca.pin_cutedsl_arch() is None
    assert dsl.envar.arch == "sm_100a"
