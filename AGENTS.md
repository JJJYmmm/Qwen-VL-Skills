# Qwen-VL Skills

This repository is a reusable agent skill package. The canonical entrypoint is `SKILL.md`; keep it concise and put longer task details in `references/`.

## Compatibility

- Codex: install or sync this folder as a skill named `qwen-vl-skills`.
- Claude Code: place this folder at `~/.claude/skills/qwen-vl-skills` or `.claude/skills/qwen-vl-skills`.
- OpenCode: place this folder at `~/.config/opencode/skills/qwen-vl-skills` or `.opencode/skills/qwen-vl-skills`. OpenCode also discovers Claude-compatible `.claude/skills` locations.

Do not add tool-specific duplicate copies of `SKILL.md`; keep one source of truth and sync/copy the folder into each tool's expected skills directory.

## Validation

Run these before publishing changes:

```bash
python -m py_compile scripts/*.py
python scripts/call_qwen_vl_api.py --dry-run --model qwen3-vl-235b-a22b-instruct --image https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg --text "Describe this image."
```

If the Codex skill validator is available, also run it against this folder.
