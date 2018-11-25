from bokeh.plotting import * # fill this in with actual functions

from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, BoxSelectTool

output_file=('scatter_totals.html')
TOOLS = 'box_zoom,box_select,crosshair,resize,reset,hover'
dt = data.groupby('date_received')['product'].count()

# TOOLS = [BoxSelectTool(), HoverTool()]
p = figure(plot_width=900, plot_height=400,x_axis_type='datetime',title='CFPB Complaints per Day')
p.circle(dt.index,dt,size=5,fill_alpha=.7,fill_color='orange',line_color=None)
p.yaxis.axis_label='Number of complaints'