# Contributing to ChaosNexus

Thank you for your interest in helping build secure, sovereign local AI infrastructure.

### Where to contribute

**Codeberg is the primary forge for issues and pull requests:** [codeberg.org/TunedChaos](https://codeberg.org/TunedChaos).

GitHub repositories under TunedChaos are **read-only mirrors** (discoverability and Sponsors). Do not open feature PRs on GitHub mirrors.

See https://chaosnexus.ai/guide/contribute and https://chaosnexus.ai/REPOSITORY_ARCHITECTURE.

### Contributor Licensing Agreement

By contributing to ChaosNexus and its associated tools, you grant Tuned Chaos a perpetual, unrestricted, worldwide license to distribute your contribution under both open-source and commercial licenses, while you retain your rights.

If you do not wish to grant these rights to Tuned Chaos, please do not submit contributions.

### Generative AI assistance

Some code in this project was generated with assistance from AI. Humans directed architecture, review, and maintenance. Read [AI_ASSISTANCE.md](AI_ASSISTANCE.md) before contributing. If you use generative tools in a pull request, disclose that you used AI assistance and keep changes small and reviewable.

## Welcome

Welcome to the ChaosNexus Contributing Guide. We are excited to have you here.

If you would like to contribute to a specific part of the project, check out the following list of contributions that we accept:
* Bug reports and fixes
* Feature requests and implementations
* Documentation improvements
* UI and UX enhancements for ChaosNexus Forge

## ChaosNexus overview

The purpose of ChaosNexus is to provide a local-first, zero-trust Agentic Orchestration Platform and IDE designed to bridge the gap between deterministic software engineering, Model Context Protocol (MCP) tool exposure, and AI agent coordination.

## Ground rules

Before contributing, please ensure you review our repository architecture and guidelines. We expect all contributors to interact respectfully and constructively. Follow the Code of Conduct.

## Community engagement

You can stay up to date with the latest news by following the Tuned Chaos dev blog at https://tunedchaos.dev.

## Environment setup

To set up your environment, ensure you have the following installed:
* Rust Toolchain (rustup, cargo, and rustc)
* Node.js and pnpm (for ChaosNexus Forge and NexusDocs)
* Python 3.10 or higher (for ChaosNexus Tuned)
* C++ Build Tools (gcc or clang)

Run the appropriate `pnpm install` or `cargo build` commands within the respective subdirectories.

## Best practices

Our project has adopted the following best practices for contributing:
* Follow the local structural guidelines and formatting styles.
* Use strict punctuation rules in documentation (no em-dashes or en-dashes, use hyphens only for word hyphenation).
* Verify changes against local test instances before submitting.
* Write clear, concise, and descriptive commit messages using semantic commit formats.

## Contribution workflow

### Branch creation
All git branches must follow semantic branch naming conventions. Please create feature branches from the `main` branch.

### Commit messages
All git commit messages must follow semantic commit message rules. This helps automate changelogs and versioning.

### Pull requests
When submitting a pull request, please ensure you complete the provided pull request checklist. Your pull request will be reviewed by the maintainers.
