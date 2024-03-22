package com.njit.ASTExample;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.ArrayList;
import java.util.List;

public class CleanStreams {

    public static void removeNullableFromStreamParameters(CompilationUnit cu) {
        NullableRemoverVisitor removerVisitor = new NullableRemoverVisitor();
        cu.accept(removerVisitor, null);
        removerVisitor.removeCollectedAnnotations();
    }

    private static class NullableRemoverVisitor extends VoidVisitorAdapter<Void> {
        private final List<AnnotationExpr> annotationsToRemove = new ArrayList<>();

        @Override
        public void visit(MethodCallExpr n, Void arg) {
            super.visit(n, arg);
            if (isStreamOperation(n)) {
                n.getArguments()
                        .forEach(
                                expr -> {
                                    expr.getChildNodesByType(Parameter.class)
                                            .forEach(this::collectNullableAnnotation);
                                });
            }
        }

        private boolean isStreamOperation(MethodCallExpr n) {
            String methodName = n.getName().asString();
            return "map".equals(methodName)
                    || "filter".equals(methodName)
                    || "flatMap".equals(methodName);
        }

        private void collectNullableAnnotation(Parameter parameter) {
            parameter.getAnnotationByName("Nullable").ifPresent(annotationsToRemove::add);
        }

        public void removeCollectedAnnotations() {
            annotationsToRemove.forEach(annotation -> annotation.remove());
        }
    }
}