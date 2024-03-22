package com.njit.ASTExample;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import java.util.ArrayList;
import java.util.List;

public class CleanDereferenced {

    public static void removeNullableOnDereferencedParams(CompilationUnit cu) {
        List<Parameter> parametersToRemoveAnnotationFrom = new ArrayList<>();

        new VoidVisitorAdapter<Void>() {
            @Override
            public void visit(MethodDeclaration md, Void arg) {
                super.visit(md, arg);

                for (Parameter param : md.getParameters()) {
                    if (hasNullableAnnotation(param) && isDereferenced(param, md)) {
                        parametersToRemoveAnnotationFrom.add(param);
                    }
                }
            }

            private boolean hasNullableAnnotation(Parameter param) {
                return param.getAnnotations().stream()
                        .anyMatch(a -> a.getNameAsString().equals("Nullable"));
            }

            private boolean isDereferenced(Parameter param, MethodDeclaration md) {
                String paramName = param.getNameAsString();
                return md.getBody().isPresent()
                        && md.getBody().get().getChildNodesByType(NameExpr.class).stream()
                                .anyMatch(nameExpr -> nameExpr.getNameAsString().equals(paramName));
            }
        }.visit(cu, null);

        // Remove the @Nullable annotations from the collected parameters after the iteration is
        // complete
        for (Parameter param : parametersToRemoveAnnotationFrom) {
            param.getAnnotations().removeIf(a -> a.getNameAsString().equals("Nullable"));
        }
    }
}
