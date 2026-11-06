r"""Converts paper/manuscript.tex to Wiley's New Journal Design v5 class (WileyNJDv5, HARVARD option) for
Earthquake Spectra, as overleaf/main.tex, with the bibliography and figures, and zips them for upload to an
Overleaf project made from Wiley's NJD v5 template (compile with XeLaTeX, TeX Live 2022)."""

import os
import re
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, 'paper', 'manuscript.tex'), encoding='utf-8').read()


def between(text, start, end):
    i = text.index(start)
    return text[i + len(start):text.index(end, i)]


abstract = between(src, r'\section{Abstract}', r'\noindent\textbf{Keywords}').strip()
body = src[src.index(r'\section{Introduction}'):src.index(r'\section{Data and Resources}')]
data = between(src, r'\section{Data and Resources}', r'\section{Acknowledgments}').strip()
acks = between(src, r'\section{Acknowledgments}', r'\bibliographystyle').strip()
captions = re.findall(r'\\caption\{(.*?)\}\n\\end\{figure\}', src, re.S)
assert len(captions) == 4, len(captions)

# Label the equations in order, and refer to them by label.
labels = ['eq:rotation', 'eq:peak', 'eq:arc', 'eq:rotd100', 'eq:rotd0', 'eq:fraction', 'eq:arias', 'eq:ariaspct',
          'eq:cav', 'eq:bound']
parts = body.split(r'\begin{equation}')
assert len(parts) == len(labels) + 1, len(parts)
body = parts[0] + ''.join(r'\begin{equation}\label{' + lab + '}' + part for lab, part in zip(labels, parts[1:]))
eq = {str(i + 1): r'\ref{' + lab + '}' for i, lab in enumerate(labels)}
figs = {'1': 'fig:hull', '2': 'fig:tabas', '3': 'fig:pacoima', '4': 'fig:sample'}


def labelled(text):
    """text with its equation and figure numbers replaced by references to their labels."""
    text = re.sub(r'Eqs\.~(\d+) to (\d+)', lambda m: f'Eqs.~{eq[m[1]]} to {eq[m[2]]}', text)
    text = re.sub(r'Eqs\.~(\d+) and (\d+)', lambda m: f'Eqs.~{eq[m[1]]} and {eq[m[2]]}', text)
    text = re.sub(r'Eq\.~(\d+)', lambda m: f'Eq.~{eq[m[1]]}', text)
    text = re.sub(r'Fig(ure|\.)~(\d)', lambda m: f'Fig{m[1]}~' + r'\ref{' + figs[m[2]] + '}', text)
    assert not re.search(r'Eqs?\.~\d|Fig(ure|\.)~\d', text), 'unconverted reference'
    return text


body = labelled(body)
captions = [labelled(c) for c in captions]

files = ['fig1_hull.pdf', 'fig2_tabas_spectra.pdf', 'fig3_pacoima.pdf', 'fig4_sample.pdf']
widths = [r'\textwidth', r'0.62\textwidth', r'\textwidth', r'\textwidth']
figures = '\n'.join(
    r'\begin{figure}[t]' + '\n' + r'\centerline{\includegraphics[width=' + w + ']{' + f[:-4] + '}}\n'
    + r'\caption{' + c.strip() + r'\label{' + lab + '}}\n' + r'\end{figure}' + '\n'
    for f, w, c, lab in zip(files, widths, captions, figs.values()))

main = r"""% Earthquake Spectra (Wiley) Technical Note, in Wiley's New Journal Design v5 class.
% Compile with XeLaTeX on TeX Live 2022 (Overleaf: Menu > Compiler XeLaTeX, TeX Live version 2022), as the
% Wiley NJD v5 template requires. Made from paper/manuscript.tex by code/make_overleaf.py.
\documentclass[HARVARD,Times1COL]{WileyNJDv5}

\makeatletter
\@ifpackageloaded{natbib}{}{\usepackage[round]{natbib}}
\makeatother
\usepackage{lineno}
\usepackage{graphicx}
% Figures are found in a figures folder or at the top level of the project.
\graphicspath{{figures/}{./}}

\newcommand{\RotD}[1]{\mathrm{RotD}#1}
\newcommand{\uvec}{\mathbf{u}}

\articletype{Technical Note}

\received{Date Month Year}
\revised{Date Month Year}
\accepted{Date Month Year}
\journal{Earthquake Spectra}
\volume{00}
\copyyear{2026}
\startpage{1}

\raggedbottom

\begin{document}

\title{Exact Computation of Orientation-Independent RotD Measures of Ground Motion Using the Convex Hull}

\author[1]{Scott J. Brandenberg}

\authormark{BRANDENBERG}
\titlemark{EXACT ROTD MEASURES USING THE CONVEX HULL}

\address[1]{\orgdiv{Department of Civil and Environmental Engineering}, \orgname{University of California, Los
Angeles}, \orgaddress{\state{California}, \country{USA}}}

\corres{Scott J. Brandenberg, Department of Civil and Environmental Engineering, University of California, Los
Angeles, Los Angeles, CA 90095, USA. \email{sjbrandenberg@ucla.edu}}

\abstract[Abstract]{""" + abstract + r"""}

\keywords{RotD50, RotD100, orientation-independent intensity measures, response spectra, convex hull, ground motion
database}

\maketitle

\linenumbers

""" + body + r"""\bmsection*{Data and Resources}
""" + data + r"""

\bmsection*{Acknowledgments}
""" + acks + r"""

\bibliography{references}

""" + figures + r"""
\end{document}
"""

out = os.path.join(ROOT, 'overleaf')
os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
open(os.path.join(out, 'main.tex'), 'w', encoding='utf-8').write(main)
shutil.copy(os.path.join(ROOT, 'paper', 'references.bib'), out)
for f in files:
    shutil.copy(os.path.join(ROOT, 'figures', f), os.path.join(out, 'figures', f))
with zipfile.ZipFile(os.path.join(ROOT, 'overleaf_upload.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
    for name in ['main.tex', 'references.bib'] + [f'figures/{f}' for f in files]:
        z.write(os.path.join(out, name), name)
print('wrote overleaf/ and overleaf_upload.zip')
