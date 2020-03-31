
objs = concept_notes tool_notes
latex = pdflatex

all: $(addsuffix .pdf,$(objs))

%.pdf : %.tex
	$(latex) $< && $(latex) $<
