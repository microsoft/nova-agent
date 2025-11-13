.ONESHELL:

SHELL := /bin/bash
# -e: Exit immediately if a command exits with a non-zero status
# -u: Treat unset variables as an error when substituting
# -o pipefail: The return value of a pipeline is the status of the last command to exit with a non-zero status
SHELLFLAGS := -eu -o pipefail -c

# Install mise-en-place and just
mise:
	@curl https://mise.run | sh
	mise install $(mise_install_args)
	if [[ "$(GITHUB_RUN_ID)" != "" ]]; then
		echo Skipping activation because running in a GitHub workflow
	else
		echo Adding activation to ~/.bashrc and ~/.zshrc
		echo 'eval "$$(~/.local/bin/mise activate bash)"' >> ~/.bashrc
		echo 'eval "$$(~/.local/bin/mise activate zsh)"' >> ~/.zshrc
		echo Open a new shell now to use mise
	fi

