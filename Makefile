# ARC Prize 2026 — ARC-AGI-3 local dev workflow.
#
# Five commands cover the whole loop:
#   make setup        # one-time: venv + arc-agi + clone framework
#   make play-local   # fast inner loop: run agent/my_agent.py on a real game
#   make pull-sample  # fetch the official Stochastic Goose sample for reference
#   make submit       # build notebook from agent/my_agent.py + push to Kaggle
#   make status       # tail the latest Kaggle run

VENV            := .venv

ifeq ($(OS),Windows_NT)
    PYTHON   ?= py -3.14
    VENV_BIN := $(VENV)/Scripts
else
    PYTHON   ?= python3.14
    VENV_BIN := $(VENV)/bin
endif

VENV_PY         := $(VENV_BIN)/python
VENV_PIP        := $(VENV_BIN)/pip
# Read the project-local token at recipe time and expose it as KAGGLE_API_TOKEN
# (the only env var the modern Kaggle CLI honours for token auth).
KAGGLE          := KAGGLE_API_TOKEN=$$(cat .kaggle/access_token) $(VENV_BIN)/kaggle
FRAMEWORK_REPO  := https://github.com/arcprize/ARC-AGI-3-Agents.git
FRAMEWORK_DIR   := vendor/ARC-AGI-3-Agents
COMP_SLUG       := arc-prize-2026-arc-agi-3
GAME            ?=
STEPS           ?= 200

.PHONY: help setup play-local pull-sample notebook submit status verify-local clean _check-kaggle

_check-kaggle:
	@$(VENV_PY) -c "import os, sys; token_file = '.kaggle/access_token'; home_json = os.path.expanduser('~/.kaggle/kaggle.json'); has_token = (os.path.exists(token_file) and os.path.getsize(token_file) > 0) or (os.path.exists(home_json) and os.path.getsize(home_json) > 0) or ('KAGGLE_API_TOKEN' in os.environ) or ('KAGGLE_USERNAME' in os.environ); sys.exit(0) if has_token else print('ERROR: Kaggle credentials missing.\n       Create an API token at https://www.kaggle.com/settings\n       and save it as .kaggle/access_token or ~/.kaggle/kaggle.json') or sys.exit(1)"

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-15s %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Vars: PYTHON=$(PYTHON)  GAME=$(GAME)  STEPS=$(STEPS)"

setup: ## One-time install: venv, arc-agi, kaggle CLI, clone framework
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install "arc-agi>=0.9.6" "kaggle>=2.2" python-dotenv pandas pyarrow
	@$(VENV_PY) -c "import os, subprocess; os.makedirs('vendor', exist_ok=True); subprocess.run(['git', 'clone', '--depth', '1', '$(FRAMEWORK_REPO)', '$(FRAMEWORK_DIR)']) if not os.path.exists('$(FRAMEWORK_DIR)/.git') else subprocess.run(['git', '-C', '$(FRAMEWORK_DIR)', 'pull', '--ff-only'])"
	@$(VENV_PY) scripts/slim_framework.py
	@echo ""
	@echo "Setup complete. Try:  make play-local"

play-local: ## Run agent/my_agent.py against ALL games (or GAME=ls20 for a single one)
	$(VENV_PY) scripts/play_local.py $(if $(GAME),--game $(GAME)) --max-steps $(STEPS)

verify-local: ## Quick smoke test: 50 steps on ls20 + vc33 only
	$(VENV_PY) scripts/play_local.py --game ls20,vc33 --max-steps 50

list-games: ## Show all available games
	$(VENV_PY) scripts/play_local.py --list

pull-sample: _check-kaggle ## Download the official Stochastic Goose sample notebook for reference
	mkdir -p reference/stochastic-goose
	$(KAGGLE) kernels pull inversion/arc3-sample-submission-stochastic-goose \
	    -p reference/stochastic-goose -m
	@echo "Open reference/stochastic-goose/*.ipynb for the canonical pattern."

notebook: ## Splice agent/my_agent.py into notebooks/submission.ipynb
	$(VENV_PY) scripts/build_notebook.py

submit: notebook _check-kaggle ## Build notebook and push to Kaggle (one-line submission)
	@$(VENV_PY) -c "import os, sys, subprocess; token_file = '.kaggle/access_token'; env = dict(os.environ); (env.update({'KAGGLE_API_TOKEN': open(token_file).read().strip()}) if os.path.exists(token_file) else None); kaggle_bin = r'$(VENV_BIN)/kaggle.exe' if os.name == 'nt' else r'$(VENV_BIN)/kaggle'; res = subprocess.run([kaggle_bin, 'kernels', 'push', '-p', 'notebooks/'], env=env); sys.exit(res.returncode)"
	@echo ""
	@echo "Pushed. Track it with:  make status"

status: _check-kaggle ## Show the status of your most recent Kaggle kernel run
	@$(VENV_PY) -c "import os, json, sys, subprocess; token_file = '.kaggle/access_token'; env = dict(os.environ); (env.update({'KAGGLE_API_TOKEN': open(token_file).read().strip()}) if os.path.exists(token_file) else None); kaggle_bin = r'$(VENV_BIN)/kaggle.exe' if os.name == 'nt' else r'$(VENV_BIN)/kaggle'; kernel_id = json.load(open('notebooks/kernel-metadata.json'))['id']; res = subprocess.run([kaggle_bin, 'kernels', 'status', kernel_id], env=env); sys.exit(res.returncode)"

clean: ## Remove generated artefacts (venv, downloaded games, vendored repos)
	rm -rf $(VENV) vendor environment_files recordings notebooks/submission.ipynb \
	       reference logs.log __pycache__ .pytest_cache
