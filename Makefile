venv:
	python -m venv ./venv
	venv/bin/python --version
	venv/bin/pip install --upgrade pip wheel
	venv/bin/pip install poetry==2.1.3

install:
	rm -f poetry.lock
	rm -f requirements.txt
	poetry install --no-root
	poetry export --output requirements.txt

test:
	PYTHONPATH=${PWD} venv/bin/coverage run -m pytest --verbose --capture=no
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
