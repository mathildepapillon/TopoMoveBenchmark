# ======================================================================
# Build the TopoSHAP paper (pdflatex + bibtex) into build/.
#   make        -> build/main.pdf
#   make clean  -> remove build/
# ======================================================================

MAIN     := main
BUILD    := build
LATEX    := pdflatex -interaction=nonstopmode -file-line-error -output-directory=$(BUILD)
SECTIONS := $(wildcard sections/*.tex)
TABLES   := $(wildcard tables/*.tex)

.PHONY: all clean

all: $(BUILD)/$(MAIN).pdf

$(BUILD)/$(MAIN).pdf: $(MAIN).tex preamble.tex results_macros.tex \
		iclr2027_conference.sty references.bib $(SECTIONS) $(TABLES)
	mkdir -p $(BUILD)
	$(LATEX) $(MAIN).tex
	# bibtex exits 1 on mere warnings (e.g. missing optional fields);
	# don't let that kill the build.
	cd $(BUILD) && BIBINPUTS=..: bibtex $(MAIN) || true
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

clean:
	rm -rf $(BUILD)
