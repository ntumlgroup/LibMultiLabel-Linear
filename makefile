# Run make to generate callgraph
# To use this, you will need to install
# 	pip install pycg
#	sudo apt install graphviz jq
SHELL := /bin/bash
PYSOURCES := $(shell find . -type f -name "*.py" -not -path "./json2dot.py")
TARGETS := nn linear

.PHONY: all
all: $(addsuffix .png, $(TARGETS))

%.png: %.dot
	dot -Tpng $< > $@

.NOTINTERMEDIATE: $(addsuffix .dot, $(TARGETS))
%.dot: %.json json2dot.py
	python3 json2dot.py $< > $@

.NOTINTERMEDIATE: $(addsuffix .json, $(TARGETS))
%.json: cg.json
	if [[ $(@:.json=) == nn ]]; then trainer=torch_trainer; else trainer=linear_trainer; fi; \
	pattern='startswith("libmultilabel.$(@:.json=)") or startswith("'$$trainer'") or startswith("main")'; \
	cat cg.json | jq 'to_entries | '\
	'[.[] | select(.key | '"$$pattern"') | '\
	'.value |= [.[] | select('"$$pattern"')] | '\
	'select(.value | length > 0)] | '\
	'from_entries' > $@

cg.json: $(PYSOURCES)
	pycg $(PYSOURCES) > cg.json

clean:
	rm -f $(addsuffix .png, $(TARGETS)) $(addsuffix .json, $(TARGETS)) $(addsuffix .dot, $(TARGETS)) cg.json

