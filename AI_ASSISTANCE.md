# AI_ASSISTANCE.md
# Disclosure of AI assistance in ChaosNexus development

**Last updated:** 2026-07-23

Tuned Chaos discloses the following in good faith, including for hosts such as [Codeberg](https://codeberg.org) whose Terms of Use (as updated July 2026) require clarity around copyright and discourage repositories that *mostly* consist of unreviewed generative-AI output.

## Summary

**Some code in this project was generated with assistance from AI.** Humans at Tuned Chaos directed the architecture, security model, product decisions, review, integration, and ongoing maintenance. This is **not** an autonomous “vibe-coded” dump, and it is **not** claimed to be free of AI assistance.

## What that means in practice

1. **Human direction:** Core design (local-first MCP host, Rhai sandbox, Forge human-in-the-loop approvals, AGPL + commercial dual licensing) is intentional product work by Tuned Chaos.
2. **Human review:** Assisted output is expected to be read, tested, and corrected before it is treated as shippable. Security-sensitive paths (permissions, sandboxing, shell/network gates) must not be accepted uncritically from any generator.
3. **Copyright and licensing:** Contributors and maintainers remain responsible for ensuring they have the right to submit and license material under AGPL-3.0-or-later (and the dual-license grant in `CONTRIBUTING.md`). Generative tools do not erase that duty.
4. **Codeberg / FLOSS hosting:** ChaosNexus is published as FLOSS under AGPL-3.0-or-later. We disclose AI assistance so visitors and moderators are not misled. We do not ask Codeberg (or any forge) to host content that exists only as unmaintained generated noise.

## Contributing with AI tools

If you use generative tools when contributing:

- Disclose that you used AI assistance in the pull request description.
- Do not submit large unreviewed generated diffs.
- Prefer small, reviewable changes.
- Never use generative tools to obscure provenance or to strip license obligations from third-party code.

See also: [CONTRIBUTING.md](CONTRIBUTING.md), [guide/ai-assistance](https://chaosnexus.ai/guide/ai-assistance).

## Contact

Questions about this disclosure: **hello@tunedchaos.com** or Codeberg issues on the relevant TunedChaos repository.
