.ONESHELL:

.PHONY: all install update test lint clean

all: install

install:
	python -m venv ./venv
	venv/bin/python --version
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install poetry==2.1.3
	@. venv/bin/activate && \
	poetry install --no-root --with dev && \
	poetry run pre-commit install && \
	poetry export --output requirements.txt --without-hashes --all-groups

update:
	poetry update --with dev
	poetry export --output requirements.txt --without-hashes --all-groups

test:
	poetry run pytest -n auto --dist loadscope --cov=. --cov-report=term --cov-report=term-missing

lint:
	poetry run ruff format
	poetry run ruff check . --fix --exit-non-zero-on-fix

clean:
	@. venv/bin/activate && \
	pre-commit uninstall && \
	rm -rf venv && \
	rm -f poetry.lock && \
	rm -f requirements.txt
