.SILENT:
.DEFAULT_GOAL := help

PROJECT := Machine Learning on Torch (Author Sebastia Agramunt)

COLOR_RESET = \033[0m
COLOR_COMMAND = \033[36m
COLOR_YELLOW = \033[33m
COLOR_GREEN = \033[32m
COLOR_RED = \033[31m

all: download-files train test-model

## Download and process files before training
download-files:
	echo "Downloading files and preprocessing data..."
	python scripts/download_files.py

## Training the model
train: 
	echo "Training model..."
	python scripts/train_model.py

test-model:
	echo "Testing model..."
	python scripts/evaluate_model.py

## Start a server
start-server:
	echo "Starting server..."
	python scripts/server.py

## Start a client with id client1
start-client:
	echo "Starting client..."
	python scripts/fake_client.py --id=client1

## Run unit test and PEP8 format
testing:
	py.test -v
	flake8
	



## Prints help message
help:
	printf "\n${COLOR_YELLOW}${PROJECT}\n------\n${COLOR_RESET}"
	awk '/^[a-zA-Z\-\_0-9\.%]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "${COLOR_COMMAND}$$ make %s${COLOR_RESET} %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST) | sort
	printf "\n"
