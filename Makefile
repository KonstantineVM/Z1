# Economic Time Series Analysis - Makefile

.PHONY: help install install-dev test test-cov lint format clean build run docker-build docker-up docker-down cache-status cache-clear docs

# Default target
help:
	@echo "Economic Time Series Analysis - Available Commands:"
	@echo "=================================================="
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black/isort"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker images"
	@echo "  make docker-up     Start all services"
	@echo "  make docker-down   Stop all services"
	@echo ""
	@echo "Cache Management:"
	@echo "  make cache-status  Show cache status"
	@echo "  make cache-clear   Clear all cache"
	@echo "  make cache-refresh Refresh cache"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Build documentation"
	@echo "  make docs-serve    Serve documentation locally"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

test-integration:
	pytest tests/integration/ -v -m integration

# Code quality
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

# Building
build: clean
	python setup.py sdist bdist_wheel

# Running
run:
	python -m src.cli.main

notebook:
	jupyter lab --port=8888

api:
	uvicorn src.api.main:app --reload --port=8000

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-shell:
	docker-compose exec app bash

# Cache management
cache-status:
	python -m src.data.cache_manager status

cache-clear:
	python -m src.data.cache_manager clear --force

cache-refresh:
	python -m src.data.cache_manager refresh

cache-clean:
	python -m src.data.cache_manager clean

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs && python -m http.server --directory _build/html 8080

docs-clean:
	cd docs && make clean

# Data management
data-download:
	python scripts/download_all_data.py

data-validate:
	python scripts/validate_data.py

# Analysis workflows
analyze-z1:
	python examples/z1_analysis.py

analyze-full:
	python examples/full_pipeline.py

# Development database
db-up:
	docker-compose --profile database up -d

db-down:
	docker-compose --profile database down

db-migrate:
	alembic upgrade head

db-reset:
	alembic downgrade base && alembic upgrade head

# Environment
env:
	cp .env.example .env

# Release
release-patch:
	bumpversion patch

release-minor:
	bumpversion minor

release-major:
	bumpversion major

# Performance profiling
profile:
	python -m cProfile -o profile.stats examples/full_pipeline.py
	python -m pstats profile.stats

# Security
security-check:
	safety check
	bandit -r src/

# Git hooks
hooks:
	pre-commit run --all-files