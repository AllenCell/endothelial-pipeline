# Contributing guidelines

### Pre-commit plugin

Use [pre-commit](https://pre-commit.com/) to automatically format your code to match the recommended coding style.
You will only need to do the following steps once.

**1. Install the development dependencies**

```bash
uv sync --only-dev --inexact
```

This command will install the development dependencies without affecting installations of optional dependency groups.

**2. Install the git hooks**

```bash
pre-commit install
```

The plugin will now run every time you commit changes.
If any of the hooks fail, or if the hooks alters files that are part of the commit, the commit will be rejected until you address the error(s) and/or stage the change(s).

If necessary, you may bypass the hooks for the commit:

```bash
git commit --no-verify -m "commit message"
```
