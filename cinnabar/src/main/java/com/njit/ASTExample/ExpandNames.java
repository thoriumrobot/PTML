package com.njit.ASTExample;

import com.github.javaparser.ast.*;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.comments.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.*;
import java.util.*;

public class ExpandNames extends ConverterSuper {

    public Map<String, Set<Integer>> nameList_old;

    public ExpandNames(Map<String, Set<Integer>> nameList) {
        this.nameList = new HashMap<>();
        this.nameList_old = new HashMap<>();

        for (Map.Entry<String, Set<Integer>> entry : nameList.entrySet()) {
            this.nameList_old.put(entry.getKey(), new HashSet<>(entry.getValue()));
        }
    }

    public void checkForOldNames(Node node, GraphNode root) {
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
            checkForOldNames(child, root);
            if (root.chooseNode) {
                return;
            }
        }
    }

    /*
    Expand the list of names by including names from other statements containing the previous list of names.
    */
    public void process(Node node, int pid) {
        int nodeId = totCount;
        String nodeType = node.getClass().getSimpleName();

        // get rid of primitive type declarations
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
        }

        // node object with id, type and boolean flag for selection (chooseNode)
        GraphNode graphNode = new GraphNode(node, nodeId, nodeType, 0);

        // mark as chosen if its subree contains one of the old names
        checkForOldNames(node, graphNode);

        // increment node count
        totCount++;

        // iterate through the child nodes
        for (Node child : node.getChildNodes()) {
            int childId = totCount;
            String childname = new String(child.getClass().getSimpleName());

            // if chosen, get the names
            if (graphNode.chooseNode && child instanceof SimpleName) {
                com.github.javaparser.ast.expr.SimpleName simpleNameNode =
                        (com.github.javaparser.ast.expr.SimpleName) child;
                String idName = simpleNameNode.asString();

                nameList.putIfAbsent(idName, new HashSet<>());
                nameList.get(idName).add(childId);
            }

            // recursively process the child if it is not an annotation subtree
            // ("MarkerAnnotationExpr")
            // if (!(child instanceof MarkerAnnotationExpr)) {
            process(child, nodeId);
            // }
        }
    }

    // wrapper
    public void convert(Node astRoot) {
        process(astRoot, 0);
    }
}
