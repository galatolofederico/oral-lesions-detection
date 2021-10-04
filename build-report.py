import argparse
import json
from fpdf import FPDF
from PIL import Image

def build_report(files, output):
    tex = r"""
    \documentclass[12pt]{article}

    \usepackage{graphicx}
    \usepackage{caption}
    \usepackage{booktabs}
    \usepackage{subfig}
    \usepackage[export]{adjustbox}
    \usepackage[margin=0.5in]{geometry}
    \usepackage{placeins}
    \usepackage{float}

    \begin{document}
    \section{Overview}

    \begin{figure}[H]
    \centering
    \subfloat[Input]{
    \includegraphics[width=0.4\textwidth,valign=c]{%s}
    }
    \subfloat[Prediction]{
    \includegraphics[width=0.4\textwidth,valign=c]{%s}        
    }
    \caption{Detection and classification}
    \end{figure}
    
    \section{Lesions}
    In the next pages you will find a detailed explanation for each leasion.
    """ % (files["input"], files["prediction"])

    for i, explain in enumerate(files["explain"]):
        classes = explain["classes"].keys()
        probs = [explain["classes"][c]["score"] for c in classes]
        
        tex += "\\newpage \n \subsection{Lesion N.%d}\n" % (i, )

        tex += r"""
        \subsubsection{Classification}
        \begin{figure}[H]
            \centering
            \subfloat[Detection Lesion N.%d]{
                \includegraphics[width=0.3\textwidth,valign=c]{%s}
            }
            \subfloat[Salicency Map Lesion N.%d]{
                \includegraphics[width=0.3\textwidth,valign=c]{%s}
            }
            \subfloat[Class Probabilities]{
                \adjustbox{valign=c}{
                \begin{tabular}{lr}
                \toprule
                \textbf{Metric} & \textbf{Score} \\ \midrule
        """ % (i, explain["lesion"], i, explain["gradcam"])

        for c, p in zip(classes, probs):
            tex += "%s & %.3f \\%%\\\\\n" % (c, 100*p)
        tex += "%s & %.3f \\%%\\\\ \\bottomrule\n" % ("healthy tissue", 100*explain["healthy_prob"])

        tex += r"""
                \end{tabular}
                }
            }
            \caption{Lesion N.%d Classification}
        \end{figure}
        

        \subsubsection{Distances}

        \begin{figure}[H]
            \centering
            \subfloat[Lesion N.%d Scatter]{
                \includegraphics[width=0.5\textwidth,valign=c]{%s}
            }
            \subfloat[Lesion N.%d Histogram]{
                \includegraphics[width=0.5\textwidth,valign=c]{%s}
            }
        \end{figure}

        \newpage
        \subsubsection{Lesion N.%d Nearest Neighbors}
        """ % (i, i, explain["scatter"], i, explain["hist"], i)


        max_class = probs.index(max(probs))
        for ic, (c, p) in enumerate(zip(classes, probs)):
            #if ic != max_class:
            #    continue
            tex += r"""
            \begin{figure}[H]
            \centering
            """
            for case in explain["classes"][c]["cases"]:
                tex += r"""
                \subfloat[dist: %.3f]{
                \includegraphics[width=0.3\textwidth,valign=c]{%s}
                }""" % (case["dist"], case["image"])

            tex += r"""
            \caption{Lesion N.%d 3-NN for %s (prob: %.3f\%%)}
            \end{figure}
                """ % (i, c, p*100)
        tex += "\\newpage"

    tex += r"\end{document}"

    report = open(output, "w")
    report.write(tex)
    report.close()


parser = argparse.ArgumentParser()

parser.add_argument("--diagnosis", type=str, required=True)
parser.add_argument("--output", type=str, default="report.tex")

args = parser.parse_args()

diagnosis = json.load(open(args.diagnosis, "r"))
build_report(diagnosis, args.output)
