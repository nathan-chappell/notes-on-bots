
mypy:
	mypy analyze.py || true

test: 
	python3 analyze.py schema_models.json || true

run:
	python3 analyze.py schema_models.json || true
	python3 analyze.py adaptive-card.json || true
