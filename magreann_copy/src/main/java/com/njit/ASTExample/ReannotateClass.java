package com.njit.ASTExample;

import com.github.javaparser.ast.*;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.comments.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.nodeTypes.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.*;
import java.lang.reflect.Field;
import java.util.*;
import java.util.HashSet;

public class ReannotateClass extends ConverterSuper {
    Node storedRoot;
    HashSet<Integer> nodeIds;
    public Map<String, Set<Integer>> nameList_old;

    public ReannotateClass(String line, Map<String, Set<Integer>> nameList) {
        String[] lines = line.split("\\n");
        nodeIds = new HashSet<>();
        this.nameList_old = new HashMap<>();

        for (int i = 0; i < lines.length; i++) {
            try {
                nodeIds.add(Integer.parseInt(lines[i]));
            } catch (NumberFormatException e) {
                System.out.println("Warning: Could not parse: " + lines[i]);
            }
        }

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
    public void process(Node node) {
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
            }
        }

        if (node instanceof ArrayType) {
            ArrayType arrayType = (ArrayType) node;
            Type arrayTypeType = arrayType.getComponentType();

            if (arrayTypeType instanceof PrimitiveType) {
                nodeType.add(arrayTypeType.toString() + "Modifier");
            } else if (arrayTypeType.toString().equals("void")) {
                nodeType.add("void");
            }
        }

        if (node instanceof ClassOrInterfaceType) {
            ClassOrInterfaceType coiType = (ClassOrInterfaceType) node;

            if (coiType.toString().equals("void")) {
                nodeType.add("void");
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
            if (nodeIds.contains(nodeId)) {
                if (node instanceof NodeWithAnnotations<?>) { // Check if node can have annotations
                    ((NodeWithAnnotations<?>) node).addAnnotation("Nullable");
                } else {
                    System.out.println(
                            "Could not annotate nodeId: " + nodeId + " of type: " + nodeType);
                }
            }

            // increment node count
            totCount++;
        }

        if (node instanceof MarkerAnnotationExpr) {
            return;
        }

        for (Node child : node.getChildNodes()) {
            boolean isNullable = false;

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
                            } // end if nullable
                        } // end going through fields of name
                    } // end if name node
                } // end iteration over children of annotation root ("MarkerAnnotationExpr")
            } // end if found annotation subtree

            if (!isNullable) {
                process(child);
            }
        }
    }

    public String toString() {
        return storedRoot.toString();
    }

    // wrapper
    public void convert(Node astRoot) {
        storedRoot = astRoot;
        process(astRoot);
    }
}
