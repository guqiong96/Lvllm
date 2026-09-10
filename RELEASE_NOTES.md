# Release Notes — v0.29.0

**Base Version:** vllm v0.29.0 (upstream) + lk_moe
**Release Type:** Feature integration release (Lvllm-v2.4.0)

## What's in this release

This release syncs the branch onto upstream vllm `v0.29.0` and keeps the **lk_moe** hybrid-inference
integration on top of it. Per the release policy the diff against upstream stays minimal — just the
lk_moe layer plus necessary bug-fix patches.

## Summary of changes

- Synced upstream vllm to `v0.29.0`.
- lk_moe CPU-GPU hybrid MoE inference (see `README.md`).
- AutoAWQ MoE: handle CPU-resident layers (keep weights off the GPU device).
- Added this `RELEASE_NOTES.md` and the portable lk_moe patch `patches/01_lk_moe__v0.29.0.patch`.

## Applying the patch

On a clean upstream `v0.29.0` checkout:

```bash
git clone --branch v0.29.0 https://github.com/vllm-project/vllm.git
cd vllm
git apply ../Lvllm/patches/01_lk_moe__v0.29.0.patch
```

## Additional support branches

New-model and architecture-specific support is provided by the following branches:

| Branch | Arch |
|--------|------|
| [Lvllmds4-x](https://github.com/guqiong96/Lvllmds4-x) | SM80+ |
| [Lvllmds4](https://github.com/guqiong96/Lvllmds4) | SM120+ |

## See also

- [Lsglang](https://github.com/guqiong96/Lsglang) — the same lk_moe integration for sglang.
- [README.md](./README.md) — full integration guide, benchmark and configuration reference.
