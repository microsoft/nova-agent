just_list:
	just --list

# Remove all virtual environments in the current folder and subfolders
rm_venvs:
	rm -rf **/.venv

# Sync all dependencies
sync_all:
	uv sync --all-groups --all-extras

# package involvement graph
pack_history PACKAGE:
	uv tree --invert --package {{PACKAGE}}

# Run type checking on the given FOLDERS (all outside submodules/ by default)
@types *FOLDERS:
	# Pyright is slower than the other tools, so this recipe gives the option to
	# run only on the folders you are interested in, e.g. `just types disco-data`.
	# This uses the base discovery Python environment but reads Pyright settings
	# from the subpackage pyproject.toml files.
	for folder in {{FOLDERS}}; do \
		echo "Running pyright for folder '$folder'..."; \
		uv run --group quality --all-extras -- pyright --project $folder; \
	done

# Run linting with ruff, with fixing of errors
lint:
	uv run --group quality -- ruff check --fix

# Run linting with ruff, displaying a diff of the changes
lint_diff:
	uv run --group quality -- ruff check --diff

# Run formatter
format_diff:
	uv run --group quality -- ruff format --diff

# Apply formatter changes
format:
	uv run --group quality -- ruff format

# Run ruff to check formatting of all files, then ruff again to find issues beyond formatting, then Pyright
check: format_diff lint_diff types

# Run unit tests on the given PATHS (all subpackages by default)
test *PATHS:
	uv run --group tests --all-extras -- pytest {{PATHS}}

submodules:
	git submodule update --init --recursive

# Update environment and install pre-commit hooks
@env: submodules
	mise install
	# We use mise exec below because the tools might not yet be in the path of the current shell
	# Install the pre-commit hooks
	# Warning, this may cause problems with the "Commit" button in VSCode
	# uv tool install pre-commit --with pre-commit-uv  # might need this
	mise exec -- uv run -- pre-commit install --install-hooks
	mise exec -- lefthook install

# Install git-lfs and pull the latest version of the repo
lfs:
	#!/usr/bin/bash
	if git-lfs > /dev/null 2>&1; then
		echo git-lfs already installed
	else
		echo Installing git-lfs
		sudo apt-get install git-lfs
	fi
	git lfs install
	git lfs pull
