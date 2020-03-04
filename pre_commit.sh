path=scripts/
# poetry run pytest -v
poetry run isort -rc ${path}
poetry run black ${path}
poetry run flake8  ${path}
