# 
# analyze schema_models.json file and produce a png with graphviz
#

python3 analyze.py > graph.gv
dot -Tpng graph.gv > graph.png

#
# now open up graph.png however you prefer and look at the results
#
