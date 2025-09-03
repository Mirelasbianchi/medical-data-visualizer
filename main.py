# main.py

# This entrypoint file is used in development. Start by reading
# and testing your functions here.

import medical_data_visualizer
from unittest import main

# Test your function by calling it here
cat_fig = medical_data_visualizer.draw_cat_plot()
cat_fig.savefig("catplot.png")   # salva o gráfico categórico

heat_fig = medical_data_visualizer.draw_heat_map()
heat_fig.savefig("heatmap.png")  # salva o heatmap

# Run unit tests automatically
main(module='test_module', exit=False)
