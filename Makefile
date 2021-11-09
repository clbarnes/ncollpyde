.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
PY_PATHS = ncollpyde tests

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr target
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find ncollpyde/ -name '*.so' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

fmt:
	isort $(PY_PATHS) \
	&& black $(PY_PATHS)
	cargo fmt

lint-python:
	black --check $(PY_PATHS)
	isort --check $(PY_PATHS)
	flake8 $(PY_PATHS)
	mypy $(PY_PATHS)

lint-rust:
	cargo fmt -- --check
	cargo clippy

lint: lint-python lint-rust

test-rust:
	cargo test

test-python: install-dev
	pytest -v --benchmark-skip

test: test-rust test-python

bench: install-opt
	pytest -v --benchmark-only

install-dev:
	maturin develop

install-opt:
	maturin develop --release

coverage: install-dev
	coverage run --source ncollpyde -m pytest && \
	coverage report -m && \
	coverage html && \
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	# rm -f docs/ncollpyde.rst && \
	# rm -f docs/modules.rst && \
	# sphinx-apidoc -o docs/ ncollpyde && \
	$(MAKE) -C docs clean && \
	$(MAKE) -C docs html

view-docs: docs
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

# release: dist ## package and upload a release
# 	twine upload dist/*

# dist: clean ## builds source and wheel package
# 	python setup.py sdist bdist_wheel && \
# 	ls -l dist
