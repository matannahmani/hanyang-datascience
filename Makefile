PYTHON = python3
SRC = apriori.py

all: run

run: input.txt
	$(PYTHON) $(SRC) 5 input.txt output.txt

clean:
	rm -f output.txt

.PHONY: all run clean
