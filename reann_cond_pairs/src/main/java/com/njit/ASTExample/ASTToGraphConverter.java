package com.njit.ASTExample;

import com.github.javaparser.ast.*;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.comments.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.expr.Name;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.*;
import java.lang.reflect.Field;
import java.util.*;
import org.json.JSONArray;
import org.json.JSONObject;

public class ASTToGraphConverter extends ConverterSuper {

    public Node storedRoot;
    public List<GraphNode> nodes;
    public Map<Integer, List<Integer>> adjacencyList;
    public Map<String, Set<Integer>> nameList_old;
    public boolean foundAnnotation;
    public int rlvCount;
    public ArrayList<Node> rlvNodes;

    public ASTToGraphConverter(Map<String, Set<Integer>> nameList) {
        this.nodes = new ArrayList<>();
        this.adjacencyList = new HashMap<>();
        this.nameList = new HashMap<>();
        this.nameList_old = new HashMap<>();
        this.foundAnnotation = false;
        this.rlvCount = 0;
        rlvNodes = new ArrayList<>();

        for (Map.Entry<String, Set<Integer>> entry : nameList.entrySet()) {
            this.nameList_old.put(entry.getKey(), new HashSet<>(entry.getValue()));
        }
    }

    public void checkForNames(Node node, GraphNode root) {
        String nodeType = node.getClass().getSimpleName();

        if (node instanceof NullLiteralExpr) {
            root.chooseNode = true;
            return;
        }

        if (node instanceof SimpleName) {
            com.github.javaparser.ast.expr.SimpleName simpleNameNode =
                    (com.github.javaparser.ast.expr.SimpleName) node;
            String idName = simpleNameNode.asString();

            if (nameList_old.containsKey(idName)) {
                root.chooseNode = true;
                return;
            }
        }

        for (Node child : node.getChildNodes()) {
            checkForNames(child, root);
            if (root.chooseNode) {
                return;
            }
        }
    }

    /*
    Prune all nodes that do not contain names in the list.
    */
    public void process(Node node, int pid) {
        int nodeId = totCount;
        ArrayList<String> nodeType = new ArrayList<>();
        nodeType.add(node.getClass().getSimpleName());

        // CatchClauses are out
        if (node instanceof CatchClause) {
            return;
        }

        if (node instanceof MarkerAnnotationExpr) {
            Set<String> stringSet = new TreeSet<>();

            for (Node child : node.getChildNodes()) {
                if (child instanceof Name) {
                    for (Field field : child.getClass().getDeclaredFields()) {
                        Name nameNode = (Name) child;
                        String identifier = nameNode.getIdentifier();

                        stringSet.add(identifier);
                    }
                }
            }

            for (String s : stringSet) {
                nodeType.add(s + "Marker");
            }
        }

        // get rid of primitive type declarations and track variable modifiers
        if (node instanceof VariableDeclarationExpr) {
            boolean skipNode = false;
            for (Node child : node.getChildNodes()) {
                if (child instanceof VariableDeclarator) {
                    for (Node grandchild : child.getChildNodes()) {
                        if (grandchild instanceof PrimitiveType) {
                            skipNode = true;
                            break;
                        }
                    }
                }
                if (skipNode) {
                    break;
                }
            }
            if (skipNode) {
                return;
            }

            VariableDeclarationExpr varDeclExpr = (VariableDeclarationExpr) node;
            Type varType = varDeclExpr.getVariable(0).getType();

            nameList.putIfAbsent(varType.toString(), new HashSet<>());
            nameList.get(varType.toString()).add(nodeId);

            for (Modifier modifier : varDeclExpr.getModifiers()) {
                nodeType.add(modifier.toString() + "Modifier");
            }
        }

        // Track return types
        if (node instanceof MethodDeclaration) {
            MethodDeclaration method = (MethodDeclaration) node;
            Type methodType = method.getType();

            if (methodType instanceof PrimitiveType) {
                nodeType.add(methodType.toString() + "Modifier");
            } else if (methodType.toString().equals("void")) {
                nodeType.add("void");
            } else {
                nameList.putIfAbsent(methodType.toString(), new HashSet<>());
                nameList.get(methodType.toString()).add(nodeId);
            }

            for (Modifier modifier : method.getModifiers()) {
                nodeType.add(modifier.toString() + "Modifier");
            }
        }

        if (node instanceof FieldDeclaration) {
            FieldDeclaration field = (FieldDeclaration) node;
            Type fieldType = field.getElementType();

            if (fieldType instanceof PrimitiveType) {
                nodeType.add(fieldType.toString() + "Modifier");
            } else if (fieldType.toString().equals("void")) {
                nodeType.add("void");
            } else {
                nameList.putIfAbsent(fieldType.toString(), new HashSet<>());
                nameList.get(fieldType.toString()).add(nodeId);
            }

            for (Modifier modifier : field.getModifiers()) {
                nodeType.add(modifier.toString() + "Modifier");
            }
        }

        if (node instanceof Parameter) {
            Parameter parameter = (Parameter) node;
            Type parameterType = parameter.getType();

            if (parameterType instanceof PrimitiveType) {
                nodeType.add(parameterType.toString() + "Modifier");
            } else if (parameterType.toString().equals("void")) {
                nodeType.add("void");
            } else {
                nameList.putIfAbsent(parameterType.toString(), new HashSet<>());
                nameList.get(parameterType.toString()).add(nodeId);
            }
        }

        if (node instanceof ArrayType) {
            ArrayType arrayType = (ArrayType) node;
            Type arrayTypeType = arrayType.getComponentType();

            if (arrayTypeType instanceof PrimitiveType) {
                nodeType.add(arrayTypeType.toString() + "Modifier");
            } else if (arrayTypeType.toString().equals("void")) {
                nodeType.add("void");
            } else {
                nameList.putIfAbsent(arrayTypeType.toString(), new HashSet<>());
                nameList.get(arrayTypeType.toString()).add(nodeId);
            }
        }

        if (node instanceof ClassOrInterfaceType) {
            ClassOrInterfaceType coiType = (ClassOrInterfaceType) node;

            if (coiType.toString().equals("void")) {
                nodeType.add("void");
            } else {
                nameList.putIfAbsent(coiType.getNameAsString(), new HashSet<>());
                nameList.get(coiType.getNameAsString()).add(nodeId);
            }
        }

        if (node instanceof ConstructorDeclaration) {
            ConstructorDeclaration constructor = (ConstructorDeclaration) node;

            for (Modifier modifier : constructor.getModifiers()) {
                nodeType.add(modifier.toString() + "Modifier");
            }
        }

        if (node instanceof EnumDeclaration) {
            EnumDeclaration enumDecl = (EnumDeclaration) node;
            String enumType = enumDecl.getName().asString();

            nameList.putIfAbsent(enumType, new HashSet<>());
            nameList.get(enumType).add(nodeId);

            for (Modifier modifier : enumDecl.getModifiers()) {
                nodeType.add(modifier.toString() + "Modifier");
            }
        }

        boolean otherFlag = false;
        for (String ind_t : nodeType) {
            if (!RECOGNIZED_TYPES_ARRAY.contains(ind_t)) {
                otherFlag = true;
            }
        }
        if (otherFlag) {
            nodeType.add("Other");
        }

        // node object with id, type and boolean flag for selection (chooseNode)
        GraphNode graphNode = new GraphNode(node, nodeId, nodeType, 0);

        // if the node doesn't contain any ExpandNames, prune it
        checkForNames(node, graphNode);
        if (!graphNode.chooseNode && !instanceInCHOSEN(node)) {
            return;
        }

        if (node instanceof MethodDeclaration) {
            MethodDeclaration method = (MethodDeclaration) node;

            for (AnnotationExpr annotation : method.getAnnotations()) {
                if ("NonNull".equals(annotation.getNameAsString())
                        || "Nonnull".equals(annotation.getNameAsString())) {
                    nodeType.add("NonNullMarker");
                }
            }
        }

        if (node instanceof FieldDeclaration) {
            FieldDeclaration field = (FieldDeclaration) node;

            for (AnnotationExpr annotation : field.getAnnotations()) {
                if ("NonNull".equals(annotation.getNameAsString())
                        || "Nonnull".equals(annotation.getNameAsString())) {
                    nodeType.add("NonNullMarker");
                }
            }
        }

        boolean rlvntFlag = false;
        for (String ind_t : nodeType) {
            if (PRIMITIVE_TYPES.contains(ind_t)) {
                rlvntFlag = false;
                break;
            }

            if (NODE_LIST.contains(ind_t)) {
                rlvntFlag = true;
            }
        }

        if (rlvntFlag) {
            rlvNodes.add(node);

            rlvCount++;
        }

        // add the node to the HashMap
        nodes.add(graphNode);

        // increment node count
        totCount++;

        if (node instanceof MarkerAnnotationExpr) {
            return;
        }

        // iterate through the child nodes
        for (Node child : node.getChildNodes()) {
            int childId = totCount;
            String childname = new String(child.getClass().getSimpleName());
            boolean isNullable = false;

            // unconditionally add every name since we pruned the irrelevant nodes (hopefully)
            if (child instanceof SimpleName) {
                com.github.javaparser.ast.expr.SimpleName simpleNameNode =
                        (com.github.javaparser.ast.expr.SimpleName) child;
                String idName = simpleNameNode.asString();

                nameList.putIfAbsent(idName, new HashSet<>());
                nameList.get(idName).add(childId);
            }

            // if annotation subtree...
            if (child instanceof MarkerAnnotationExpr) {
                // iterate through children
                for (Node grandchild : child.getChildNodes()) {
                    String grandchildname = new String(grandchild.getClass().getSimpleName());
                    // if we're seeing the annotation name
                    if (grandchild instanceof Name) {
                        for (Field field : grandchild.getClass().getDeclaredFields()) {
                            Name nameNode = (Name) grandchild;
                            String identifier = nameNode.getIdentifier();

                            // if it's a nullable annotation, mark the node as nullable
                            if (identifier.equals("Nullable")) {
                                isNullable = true;
                                this.foundAnnotation = true;
                                GraphNode last = nodes.get(nodes.size() - 1);
                                last.nullable = 1;
                            } // end if nullable
                        } // end going through fields of name
                    } // end if name node
                } // end iteration over children of annotation root ("MarkerAnnotationExpr")
            } // end if found annotation subtree

            // if (!isNullable) {
            // update neighbors list
            if (graphNode.chooseNode) {
                adjacencyList.putIfAbsent(nodeId, new ArrayList<>());
                adjacencyList.get(nodeId).add(childId);
            } else {
                adjacencyList.putIfAbsent(pid, new ArrayList<>());
                adjacencyList.get(pid).add(childId);
            }

            // recursively process the child if it is not an annotation subtree
            // ("MarkerAnnotationExpr")
            process(child, nodeId);
            // }
        }
    }

    public List<GraphNode> getNodes() {
        return nodes;
    }

    public Map<Integer, List<Integer>> getAdjacencyList() {
        return adjacencyList;
    }

    // wrapper
    public void convert(Node astRoot) {
        storedRoot = astRoot;
        process(astRoot, 0);
    }

    public JSONObject toJson() {
        JSONArray nodesJsonArray = new JSONArray();
        for (GraphNode node : nodes) {
            JSONObject nodeJson = new JSONObject();
            nodeJson.put("id", node.id);
            // Convert node.type (ArrayList<String>) to JSONArray
            JSONArray typeJsonArray = new JSONArray(node.type);
            nodeJson.put("type", typeJsonArray);
            nodeJson.put("nullable", node.nullable);
            nodesJsonArray.put(nodeJson);
        }

        JSONObject adjacencyListJson = new JSONObject();
        for (Map.Entry<Integer, List<Integer>> entry : adjacencyList.entrySet()) {
            adjacencyListJson.put(entry.getKey().toString(), new JSONArray(entry.getValue()));
        }

        JSONObject nameListJson = new JSONObject();
        for (Map.Entry<String, Set<Integer>> entry : nameList.entrySet()) {
            nameListJson.put(entry.getKey().toString(), new JSONArray(entry.getValue()));
        }

        JSONObject graphJson = new JSONObject();
        graphJson.put("nodes", nodesJsonArray);
        graphJson.put("adjacencyList", adjacencyListJson);
        graphJson.put("nameList", nameListJson);

        return graphJson;
    }
}
