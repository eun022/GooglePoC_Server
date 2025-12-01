from .bar import bar_single_highlight
from .line import line_single_highlight
from .mixed import mixed_single_highlight
from .boxplot import boxplot_single_highlight
from .violin import violin_single_highlight
from .pie import pie_single_highlight
from .treemap import treemap_single_highlight

def get_draw():
    QA_mapping = {
        "bar": bar_single_highlight,
        "scatter": bar_single_highlight,
        "line": line_single_highlight,
        "pie": pie_single_highlight,
        "boxplot": boxplot_single_highlight,
        "violin": violin_single_highlight,
        "treemap": treemap_single_highlight,
        "mixed": mixed_single_highlight,
    }
    return QA_mapping