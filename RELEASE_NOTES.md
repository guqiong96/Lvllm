# Release Notes — 3116c5d

**Base Version:** vllm commit 3116c5d (upstream, incl. UVA PLE-offload + Engram TP) + lk_moe
**Release Type:** Feature integration release (Lvllm)

## What's in this release

This release is rebuilt onto upstream vllm base commit `3116c5d` (which includes the official
`[Qwen4Exp] Support UVA PLE-offload and Engram tensor parallelism` PR) and keeps the **lk_moe**
hybrid-inference integration on top of it. Per the release policy the diff against upstream stays
minimal — just the lk_moe layer plus necessary bug-fix patches.

## Summary of changes

- Rebuilt onto upstream base commit `3116c5d` (UVA PLE-offload + Engram TP).
- lk_moe CPU-GPU hybrid MoE inference (see `README.md`).
- AutoAWQ MoE: handle CPU-resident layers (keep weights off the GPU device).
- Added this `RELEASE_NOTES.md` and the portable lk_moe patch `patches/01_lk_moe__3116c5d.patch`.

## Applying the patch

On a clean upstream `3116c5d` checkout:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 3116c5d
git apply ../Lvllm/patches/01_lk_moe__3116c5d.patch
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
