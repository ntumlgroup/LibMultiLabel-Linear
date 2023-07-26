SHELL := /bin/bash
all: init activate

## Customize your python if needed
#PYTHON := $(HOME)/miniconda3/bin/python3.9
PYTHON := /usr/bin/python3

init:
	$(PYTHON) -m venv ./venv
	chmod +x venv/bin/activate
activate:
	source ./venv/bin/activate; \
	cat requirements.txt | xargs -n 1 -L 1 pip install; \

clean:
	rm -rf venv
