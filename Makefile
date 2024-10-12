PROJECT_NAME=latentspace-explorer
PACKAGE_NAME=lse


# Python env
PYTHON_PROJECT_VERSION := $(shell cat .python-version | tr -d '[:space:]')
PYTHON_SHELL_VERSION := $(shell python --version | cut -d " " -f 2)
POETRY_AVAILABLE := $(shell which poetry > /dev/null && echo 1 || echo 0)

# CI variables
CI_EXCLUDED_DIRS = __pycache__ dist tests docs demo server
CI_DIRECTORIES=$(filter-out $(CI_EXCLUDED_DIRS), $(foreach dir, $(dir $(wildcard */)), $(dir:/=)))

# Container variables
PYTHON_DOCKER_IMAGE=python:${PYTHON_PROJECT_VERSION}-slim
PRODUCTION_TAG= $(shell cat .production-tag | tr -d '[:space:]')
APP_DOCKER_IMAGE=$(PROJECT_NAME)-app
#ARTIFACT_REGISTRY=parici75
ARTIFACT_REGISTRY=us-central1-docker.pkg.dev/latentspaceexplorer/$(PROJECT_NAME)
CONTAINER_PORT=8050
PROD_WORKERS=4

# Project targets
confirm:
	@echo "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]

exists-%:
	@which "$*" > /dev/null && echo 1 || echo 0

config: check-python-env
	@cp -n .env.sample .env || true

print-%: ; @echo $* = $($*)

check-python-env:
	@if [ "$(PYTHON_PROJECT_VERSION)" == "" ] || [ "$(PYTHON_PROJECT_VERSION)" != "$(PYTHON_SHELL_VERSION)" ]; \
		then echo "The PYTHON_VERSION env variable must be set in .python-version and must be the same as the local Python environment before running this command" && exit 1;\
	fi

check-production-env: config
	@if [ "$(PRODUCTION_TAG)" == "" ]; \
		then echo "The PRODUCTION_TAG env variable must be set in .production-tag" before running this command && exit 1;\
	fi

init:
ifneq ($(POETRY_AVAILABLE), 1)
	@echo "No Poetry executable found, cannot init project" && exit 1;
endif
	@poetry check --no-ansi --quiet
	@echo "✅ Poetry is installed"
	@echo "💡 Using Python $(PYTHON_SHELL_VERSION)"
	@poetry config virtualenvs.in-project true
	@poetry config virtualenvs.create true
	@poetry install

	pre-commit

# CI targets
lint-%:
	@echo lint-"$*"
	@poetry run black --check "$*"
	@poetry run isort --check "$*"
	@poetry run ruff check "$*"
	@echo "    ✅ All good"

lint: $(addprefix lint-, $(CI_DIRECTORIES))

typecheck-%:
	@echo typecheck-"$*"
	@poetry run mypy "$*"

typecheck: $(addprefix typecheck-, $(CI_DIRECTORIES))

test:
	@poetry run pytest -s -o log-cli=true --rootdir ./  --cache-clear tests

ci: lint typecheck test

coverage:
	@poetry run coverage run -m pytest
	@poetry run coverage report

# Pre-commit hooks
set-pre-commit:
	@echo "Setting up pre-commit hooks..."
	@poetry run pre-commit install
	@poetry run pre-commit autoupdate

run-pre-commit:
	@poetry run pre-commit run --all-files

pre-commit: set-pre-commit run-pre-commit

# Documentation
update-doc:
	@poetry run sphinx-apidoc --module-first --no-toc --force -o docs/source $(PACKAGE_NAME)


build-doc:
	@poetry run python docs/export_notebook_attachments.py
	@poetry run sphinx-build docs ./docs/_build/html/


# Build dev App
build-app: check-python-env
	@echo "Building application image: $(APP_DOCKER_IMAGE)"
	@docker buildx bake -f docker-compose.yaml \
		--set app.args.PYTHON_DOCKER_IMAGE=$(PYTHON_DOCKER_IMAGE) \

run-app: config build-app
	@docker compose up --force-recreate

start-redis: config
	@docker run -p 6379:6379 redis


# Build and Push production container
build-prod-container: check-production-env
	@echo "Building production image: $(APP_DOCKER_IMAGE):$(PRODUCTION_TAG)"
	@docker build --platform=linux/amd64 \
	--build-arg PYTHON_DOCKER_IMAGE=$(PYTHON_DOCKER_IMAGE) \
	--build-arg CONTAINER_PORT=$(CONTAINER_PORT) \
	--build-arg WORKERS=$(PROD_WORKERS) \
	-t $(ARTIFACT_REGISTRY)/$(APP_DOCKER_IMAGE):$(PRODUCTION_TAG) \
	.

push-prod-container: build-prod-container
	@docker push $(ARTIFACT_REGISTRY)/$(APP_DOCKER_IMAGE):$(PRODUCTION_TAG)

run-prod-container: config build-prod-container
	@docker run -p $(CONTAINER_PORT):$(CONTAINER_PORT) --env-file .env --rm $(ARTIFACT_REGISTRY)/$(APP_DOCKER_IMAGE):$(PRODUCTION_TAG) --log-level=debug

# Cleaning
clean-python:
	@echo "🧹 Cleaning Python bytecode..."
	@poetry run pyclean . --quiet

clean-cache:
	@echo "🧹 Cleaning cache..."
	@find . -regex ".*_cache" -type d -print0|xargs -0 rm -r --
	@poetry run pre-commit clean

clean-hooks:
	@echo "🧹 Cleaning hooks..."
	@rm -r ".git/hooks" ||:


# Global
clean: confirm clean-cache clean-python clean-hooks
	@echo "✨ All clean"
