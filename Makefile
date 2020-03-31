
objs = concept_notes tool_notes mini_projects
latex = pdflatex

all: $(addsuffix .pdf,$(objs))

%.pdf : %.tex
	$(latex) $< && $(latex) $<
