# LaTeX source (ACM SIGPLAN, LCTES full-paper format)

- `main.tex`, `references.bib`, `figures/prune_sweep.png` → build with `latexmk -pdf main.tex`
  (or upload `paper_latex.zip` to Overleaf; compiler pdfLaTeX).
- Submission options: `\documentclass[sigplan,10pt,review,anonymous,nonacm]{acmart}` (anonymous, line numbers).
  Camera-ready: remove `review,anonymous,nonacm` and add the publisher's rights block.
- Page budget (LCTES full papers): 10 pages + up to 2 pages of references/appendix. Current build: body on pages 1–9,
  references and appendix end on page 10.
- Open item: on-device ESP32 latency (Section 4.9, marked in bold).
- Where every number comes from: `PROVENANCE.md` (also Appendix D of the paper).
