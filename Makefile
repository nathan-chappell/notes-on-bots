
objs = notes bot_framework_notes
latex = pdflatex

all: $(addsuffix .pdf,$(objs))

%.pdf : %.tex
	$(latex) $< && $(latex) $<
