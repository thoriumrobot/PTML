import json
import numpy as np
import os
from json_tricks import loads
import glob

def getobj(file):
   while True:
    s=file.read(1)
    if not s:
        return s
    if s=='{':
       break
   depth=1
   while depth>0:
      char=file.read(1)
      if char=='{':
         depth+=1
      if char=='}':
         depth-=1
      s+=char
   return s

directory = '/home/k/ks225/annotations/2hv/'
files = glob.glob(os.path.join(directory, '*' + '.json'))

types=dict()
primitive_types=["void", "byte", "short", "int", "long", "float", "double", "char", "boolean"]

for fname in files:
  print(fname)

  # Load JSON data from file
  with open(fname, "r") as file:
      while True:
          obj_str = getobj(file)
          if not obj_str:
            break
          obj_str = loads(obj_str)
          nodes=obj_str['nodes']
          for node in nodes:
            for type in node['type']:
               types[type] = types.get(type, 0) + 1

nodeList=["MethodDeclaration", "Parameter", "FieldDeclaration", "ArrayType", "ClassOrInterfaceType", "VariableDeclarationExpr"]

types_list = [k for k, v in types.items() if v > 5]
'''
l3= [x for x in l2 if any(y in x for y in l1) and x not in primitive_types]
types=[x for x in l2 if '*' not in x and '/' not in x]
nodeList=[x for x in l3 if '*' not in x and '/' not in x]
'''

print(len(primitive_types))

print("nodeList=",primitive_types)

print(len(nodeList))

print("nodeList=",nodeList)

print(len(types_list))

print("types=",types_list)
