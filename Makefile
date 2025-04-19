venv: requirements.txt
	python3 -m venv ./venv
	venv/bin/pip install --upgrade pip wheel
	venv/bin/pip install --upgrade -r requirements.txt

test:
	PYTHONPATH=${PWD} venv/bin/coverage run -m pytest --verbose --capture=no
	PYTHONPATH=${PWD} venv/bin/coverage report

lint:
	venv/bin/ruff format .
	venv/bin/ruff check . --fix --exit-non-zero-on-fix

clean:
	rm -rf venv
