package com.njit.ASTExample;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class NullableAnnotator {

    public static void annotateParameters(CompilationUnit cu) {
        cu.accept(new MethodVisitor(), null);
    }

    private static class MethodVisitor extends VoidVisitorAdapter<Void> {
        @Override
        public void visit(MethodDeclaration md, Void arg) {
            for (Parameter param : md.getParameters()) {
                String paramName = param.getNameAsString();
                if (methodAssignsParamToNull(md, paramName)) {
                    param.addAnnotation("Nullable");
                }
            }
        }

        private boolean methodAssignsParamToNull(MethodDeclaration md, String paramName) {
            AssignToNullVisitor assignToNullVisitor = new AssignToNullVisitor(paramName);
            md.accept(assignToNullVisitor, null);
            return assignToNullVisitor.isAssignedToNull();
        }
    }

    private static class AssignToNullVisitor extends VoidVisitorAdapter<Void> {
        private final String paramName;
        private boolean assignedToNull = false;

        public AssignToNullVisitor(String paramName) {
            this.paramName = paramName;
        }

        @Override
        public void visit(AssignExpr ae, Void arg) {
            if (ae.getTarget() instanceof NameExpr) {
                NameExpr nameExpr = (NameExpr) ae.getTarget();
                if (nameExpr.getNameAsString().equals(paramName)
                        && ae.getValue() instanceof NullLiteralExpr) {
                    assignedToNull = true;
                }
            }
        }

        public boolean isAssignedToNull() {
            return assignedToNull;
        }
    }
}
