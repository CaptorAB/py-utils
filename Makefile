venv:
	python -m venv ./venv
	venv/bin/python --version
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install poetry==2.1.3

install:
	rm -f poetry.lock
	rm -f requirements.txt
	poetry install --no-root --with dev
	poetry run pre-commit install
	poetry export --output requirements.txt --without-hashes --all-groups

update:
	poetry update
	poetry export --output requirements.txt --without-hashes --all-groups

test:
	PYTHONPATH=${PWD} venv/bin/coverage run -m pytest --verbose
	venv/bin/coverage xml --quiet
	venv/bin/coverage report
	venv/bin/genbadge coverage --silent --local --input-file coverage.xml --output-file coverage.svg

lint:
	venv/bin/ruff format
	venv/bin/ruff check . --fix --exit-non-zero-on-fix

clean:
	rm -rf venv
	rm -f poetry.lock
	rm -f requirements.txt
