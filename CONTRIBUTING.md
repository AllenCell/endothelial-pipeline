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

### Multiple virtual environments

If you want to run workflows on different systems with code stored at the same location, we recommend defining dedicated virtual environments for each system.
This will ensure that `uv` correctly installs packages and, if needed, links them to system libraries.

By default `uv` will create the virtual environment for the project at `.venv`.
Instead, for each different system, specify the `UV_PROJECT_ENVIRONMENT` environment variable before syncing virtual environment.
For example:

**Slurm**

```bash
export UV_PROJECT_ENVIRONMENT=.venv_slurm
uv sync
```

**A100s**

```bash
export UV_PROJECT_ENVIRONMENT=.venv_a100s
uv sync
```

This creates two separate environments, one for a Slurm cluster and one for A100s machines.
You can then activate the appropriate environment, or use `uv run`, which will use the environment specified by `UV_PROJECT_ENVIRONMENT`.
You can check which environment `uv` is using with:

```bash
echo $UV_PROJECT_ENVIRONMENT
```

If no value is returned, `uv` will default to using `.venv`.

If you often switch between systems, it may be helpful to add the `export` statement in your `.bashrc` (or equivalent) shell configuration file to automatically set the environment variable.
