.PHONY: test lint bootstrap run

test:
	pytest -q

lint:
	python -m compileall api config data scripts strategy tests

bootstrap:
	python scripts/bootstrap.py --check

run:
	uvicorn api.main:app --reload
