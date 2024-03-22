import json
import numpy as np
from scipy.sparse import csr_matrix
import os
import subprocess

directory = '/home/k/ks225/restricted/ensemble/JavaParser_comb/'
nodeList=["MethodDeclaration", "Parameter", "FieldDeclaration", "ArrayType", "ClassOrInterfaceType", "VariableDeclarationExpr"]

for nodeType in nodeList:
    fname=os.path.join(directory, "output_"+nodeType+".json")

    print(fname)
    
    json_data = []
    
    # Load JSON data from file
    with open(fname, "r") as file:
      json_data = json.load(file)
    
    print(len(json_data))
