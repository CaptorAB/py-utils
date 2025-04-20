venv: requirements.txt
	python3 -m venv ./venv
	venv/bin/pip install --upgrade pip wheel
	venv/bin/pip install --upgrade -r requirements.txt

test:
	PYTHONPATH=${PWD} venv/bin/coverage run -m pytest --verbose --capture=no
	venv/bin/coverage xml --quiet
	venv/bin/coverage report
	venv/bin/genbadge coverage --silent --local --input-file coverage.xml --output-file coverage.svg

lint:
	venv/bin/ruff format .
	venv/bin/ruff check . --fix --exit-non-zero-on-fix

clean:
	rm -rf venv
