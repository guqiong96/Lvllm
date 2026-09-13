# Release Notes — 71888f507a

**Base Version:** vllm commit 71888f507a (`upstream/main`, 2026-09-14) + lk_moe
**Release Type:** Feature integration release (Lvllm)

## What's in this release

This release is rebuilt onto upstream vllm base commit `71888f507a` (`upstream/main` as of
2026-09-14) and keeps the **lk_moe**
hybrid-inference integration on top of it. Per the release policy the diff against upstream stays
minimal — just the lk_moe layer plus necessary bug-fix patches.

## Summary of changes

- Synced upstream base `3116c5d` -> `71888f507a` (+204 commits: DeepSeek-V4.1, Qwen3.8-Flash-Next, GLM5-next, etc.); lk_moe re-applied.
- lk_moe CPU-GPU hybrid MoE inference (see `README.md`).
- AutoAWQ MoE: handle CPU-resident layers (keep weights off the GPU device).
- Added this `RELEASE_NOTES.md` and the portable lk_moe patch `patches/01_lk_moe__71888f507a.patch`.

## Applying the patch

On a clean upstream `71888f507a` checkout:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 71888f507a
git apply ../Lvllm/patches/01_lk_moe__71888f507a.patch
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
