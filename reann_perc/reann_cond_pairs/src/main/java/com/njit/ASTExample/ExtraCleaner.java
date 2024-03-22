package com.njit.ASTExample;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.nodeTypes.NodeWithAnnotations;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import java.util.*;

public class ExtraCleaner {

    public static void removeExtraNullableAnnotations(CompilationUnit cu) {
        cu.accept(
                new VoidVisitorAdapter<Void>() {
                    @Override
                    public void visit(FieldDeclaration n, Void arg) {
                        super.visit(n, arg);
                        removeExtraNullable(n);
                    }

                    @Override
                    public void visit(MethodDeclaration n, Void arg) {
                        super.visit(n, arg);
                        removeExtraNullable(n);
                        // Also handle return type
                        removeExtraNullable((NodeWithAnnotations<?>) n.getType());
                        // Handle method parameters
                        for (Parameter param : n.getParameters()) {
                            removeExtraNullable((NodeWithAnnotations<?>) param.getType());
                        }
                    }

                    @Override
                    public void visit(Parameter n, Void arg) {
                        super.visit(n, arg);
                        removeExtraNullable(n);
                    }

                    private void removeExtraNullable(NodeWithAnnotations<?> node) {
                        List<AnnotationExpr> toRemove = new ArrayList<>();
                        long count =
                                node.getAnnotations().stream()
                                        .filter(a -> a.getNameAsString().equals("Nullable"))
                                        .count();

                        if (count > 1) {
                            boolean removedOnce = false;
                            for (AnnotationExpr annotation : node.getAnnotations()) {
                                if (annotation.getNameAsString().equals("Nullable")) {
                                    if (!removedOnce) {
                                        toRemove.add(annotation);
                                        removedOnce = true;
                                    }
                                }
                            }
                            toRemove.forEach(AnnotationExpr::remove);
                        }
                    }
                },
                null);
    }
}
