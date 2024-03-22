package com.njit.ASTExample;

import com.github.javaparser.ast.*;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.comments.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.*;
import java.util.*;

public class BaseNames extends ConverterSuper {

    public BaseNames() {
        this.nameList = new HashMap<>();
    }

    public void checkForNull(Node node, GraphNode root) {
        if (node instanceof NullLiteralExpr) {
            root.chooseNode = true;
            return;
        }

        for (Node child : node.getChildNodes()) {
            checkForNull(child, root);
            if (root.chooseNode) {
                return;
            }
        }
    }

    /*
    Traverses the AST, finding names inside possibly nullable nodes.
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

        // if current node is possibly nullable or ranked high in the ablation study, its names will
        // be added to nameList (flag chooseNode is true)
        if (instanceInNODE(node) || instanceInCHOSEN(node)) {
            graphNode.chooseNode = true;
        }

        // if the node is a statement and there are node types chosen by ablation in its subtree,
        // its names will also be added to nameList
        if (instanceInSTMT(node)) {
            checkForNull(node, graphNode);
        }

        // increment node count
        totCount++;

        // iterate through the child nodes
        for (Node child : node.getChildNodes()) {
            int childId = totCount;
            String childname = new String(child.getClass().getSimpleName());

            // if node is chosen, get the names
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
