package com.njit.ASTExample;

import com.github.javaparser.*;
import com.github.javaparser.ast.*;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.comments.*;
import com.github.javaparser.ast.nodeTypes.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Stream;

public class NullableProcessorByName {

    public static void nPBM(String projectPath) {
        // Iterate over all .java files in the project
        try (Stream<Path> paths = Files.walk(Paths.get(projectPath))) {
            paths.filter(Files::isRegularFile)
                 .filter(p -> p.toString().endsWith(".java"))
                 .forEach(NullableProcessorByName::processFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void processFile(Path javaFile) {
        try {
            JavaParser parser = new JavaParser();
            CompilationUnit cu = parser.parse(javaFile).getResult().orElse(null);

            if (cu != null) {
                cu.accept(new MethodVisitor(), null);

                        // sanity check conditions

                        // assigned to null
                        annotateFieldsAssignedToNull(cu);

                        // synchronize Methods with Fields
                        syncMethodField(cu);

                        // assignment to Null
                        NullableAnnotator.annotateParameters(cu);

                        // comparisons with null for Fields
                        markNullable(cu);

                        // comparisons with null for Parameters
                        markNullableParameters(cu);

                        // synchronize Parameters with Fields
                        annotateParametersAsNullable(cu);
                        
                        //returns null
                        addNullableAnnotation(cu);

                        // remove extras
                        ExtraCleaner.removeExtraNullableAnnotations(cu);

                        // remove object creation
                        removeNullableFromNewConstructors(cu);

                        // remove annotations from stream Parameters
                        CleanStreams.removeNullableFromStreamParameters(cu);

                        // remove annotations from dereferenced Parameters
                        CleanDereferenced.removeNullableOnDereferencedParams(cu);
                                        
                                // remove annotations from constant methods
                                removeNullableFromConstantMethods(cu);
                                
                                File file=javaFile.toFile();

                                String importline = null;

                                if (file.toString().contains("keyvaluestore"))
                                    importline = "io.reactivex.annotations.Nullable";
                                else if (file.toString().contains("meal-planner"))
                                    importline = "javax.annotation.Nullable";
                                else if (file.toString().contains("QRContact"))
                                    importline = "android.support.annotation.Nullable";
                                else if (file.toString().contains("caffeine"))
                                    importline =
                                            "org.checkerframework.checker.nullness.qual.Nullable";
                                else if (file.toString().contains("ColdSnap"))
                                    importline = "javax.annotation.Nullable";
                                else if (file.toString().contains("AutoDispose"))
                                    importline = "io.reactivex.annotations.Nullable";
                                else if (file.toString().contains("okbuck"))
                                    importline = "javax.annotation.Nullable";
                                else if (file.toString().contains("picasso"))
                                    importline = "androidx.annotation.Nullable";
                                else if (file.toString().contains("ReactiveNetwork"))
                                    importline = "javax.annotation.Nullable";
                                else if (file.toString().contains("skaffold-tools-for-java"))
                                    importline = "javax.annotation.Nullable";
                                else if (file.toString().contains("butterknife"))
                                    importline = "android.support.annotation.Nullable";
                                else if (file.toString().contains("uLeak"))
                                    importline = "android.support.annotation.Nullable";
                                else if (file.toString().contains("RIBs"))
                                    importline = "org.jetbrains.annotations.Nullable";
                                else if (file.toString().contains("jib"))
                                    importline = "javax.annotation.Nullable";

                                if (importline != null) {
                                    cu.addImport(importline);
                                }

                // Write modifications back to the file
                Files.write(javaFile, cu.toString().getBytes());
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void annotateFieldsAssignedToNull(CompilationUnit cu) {
        for (FieldDeclaration fieldDeclaration : cu.findAll(FieldDeclaration.class)) {
            for (VariableDeclarator variable : fieldDeclaration.getVariables()) {
                if (variable.getInitializer().isPresent()
                        && variable.getInitializer().get() instanceof NullLiteralExpr) {
                    // Check if @Nullable annotation is already present
                    boolean nullableAnnotationPresent = false;
                    for (AnnotationExpr annotation : fieldDeclaration.getAnnotations()) {
                        if (annotation.getNameAsString().equals("Nullable")) {
                            nullableAnnotationPresent = true;
                            break;
                        }
                    }
                    // If not present, add the annotation
                    if (!nullableAnnotationPresent) {
                        fieldDeclaration.addAnnotation(new MarkerAnnotationExpr("Nullable"));
                    }
                }
            }
        }
    }

    public static void syncMethodField(CompilationUnit cu) {
        for (FieldDeclaration field : cu.findAll(FieldDeclaration.class)) {
            // Check if the field has a @Nullable annotation
            if (field.getAnnotationByName("Nullable").isPresent()) {
                String fieldName = field.getVariable(0).getNameAsString();

                for (MethodDeclaration method : cu.findAll(MethodDeclaration.class)) {
                    // Check if method mentions the field in its return expression
                    if (mentionsFieldInReturn(method, fieldName)) {
                        // If the method does not have a @Nullable annotation, add it
                        if (!method.getAnnotationByName("Nullable").isPresent()) {
                            method.addAnnotation("Nullable");
                        }
                    }
                }
            }
        }
    }

    public static boolean mentionsFieldInReturn(MethodDeclaration method, String fieldName) {
        for (ReturnStmt returnStmt : method.findAll(ReturnStmt.class)) {
            if (returnStmt.getExpression().isPresent()) {
                Expression returnExpr = returnStmt.getExpression().get();
                // Check if the return expression mentions the field name anywhere
                if (returnExpr
                                .findAll(
                                        NameExpr.class,
                                        expr -> expr.getNameAsString().equals(fieldName))
                                .size()
                        > 0) {
                    return true;
                }
            }
        }
        return false;
    }

    public static void markNullable(CompilationUnit cu) {
        Set<String> nullableNames = new HashSet<>();

        // Collect names of fields that are checked against null
        cu.accept(
                new VoidVisitorAdapter<Void>() {
                    @Override
                    public void visit(BinaryExpr n, Void arg) {
                        if (n.getOperator() == BinaryExpr.Operator.EQUALS
                                || n.getOperator() == BinaryExpr.Operator.NOT_EQUALS) {
                            if (n.getLeft() instanceof NameExpr
                                    && n.getRight() instanceof NullLiteralExpr) {
                                nullableNames.add(((NameExpr) n.getLeft()).getNameAsString());
                            } else if (n.getRight() instanceof NameExpr
                                    && n.getLeft() instanceof NullLiteralExpr) {
                                nullableNames.add(((NameExpr) n.getRight()).getNameAsString());
                            }
                        }
                        super.visit(n, arg);
                    }
                },
                null);

        // Annotate fields that were found to be checked against null
        cu.accept(
                new VoidVisitorAdapter<Void>() {
                    @Override
                    public void visit(FieldDeclaration n, Void arg) {
                        if (nullableNames.contains(n.getVariables().get(0).getNameAsString())) {
                            n.addMarkerAnnotation("Nullable");
                        }
                        super.visit(n, arg);
                    }
                },
                null);
    }

    public static void markNullableParameters(CompilationUnit cu) {
        for (MethodDeclaration method : cu.findAll(MethodDeclaration.class)) {
            for (BinaryExpr binaryExpr : method.findAll(BinaryExpr.class)) {
                // Check if the expression is a comparison against null
                if ((binaryExpr.getOperator() == BinaryExpr.Operator.EQUALS
                                || binaryExpr.getOperator() == BinaryExpr.Operator.NOT_EQUALS)
                        && (binaryExpr.getLeft() instanceof NameExpr
                                || binaryExpr.getRight() instanceof NameExpr)) {
                    NameExpr nameExpr = null;
                    if (binaryExpr.getLeft() instanceof NullLiteralExpr) {
                        nameExpr = (NameExpr) binaryExpr.getRight();
                    } else if (binaryExpr.getRight() instanceof NullLiteralExpr) {
                        nameExpr = (NameExpr) binaryExpr.getLeft();
                    }

                    if (nameExpr != null) {
                        // Check if the NameExpr is a parameter of the method
                        for (Parameter param : method.getParameters()) {
                            if (param.getName().asString().equals(nameExpr.getNameAsString())) {
                                // Add @Nullable annotation
                                param.addAnnotation("Nullable");
                            }
                        }
                    }
                }
            }
        }
    }

    public static void annotateParametersAsNullable(CompilationUnit cu) {
        // 1. Parse the CompilationUnit.

        // 2. Find all the FieldDeclaration nodes with the @Nullable annotation.
        Set<String> nullableFields = new HashSet<>();
        for (FieldDeclaration field : cu.findAll(FieldDeclaration.class)) {
            for (AnnotationExpr annotation : field.getAnnotations()) {
                if (annotation.getNameAsString().equals("Nullable")) {
                    nullableFields.add(field.getVariable(0).getNameAsString());
                }
            }
        }

        // 3. For each such field, collect the field's name.

        // 4. Search for all methods in the CompilationUnit.
        for (MethodDeclaration method : cu.findAll(MethodDeclaration.class)) {
            // 5. For each method, search for parameters whose name matches one of the collected
            // field names.
            for (Parameter parameter : method.getParameters()) {
                if (nullableFields.contains(parameter.getNameAsString())) {
                    // 6. Annotate those parameters with @Nullable.
                    parameter.addAnnotation(new MarkerAnnotationExpr("Nullable"));
                }
            }
        }
    }

    public static void addNullableAnnotation(CompilationUnit cu) {
        cu.accept(new VoidVisitorAdapter<Void>() {
            @Override
            public void visit(MethodDeclaration md, Void arg) {
                super.visit(md, arg);
                
                // Check if method already has @Nullable annotation
                if (md.getAnnotationByName("Nullable").isPresent()) {
                    return;
                }
                
                // Check if any return statement returns null or has a ternary with null
                boolean shouldAnnotate = md.getBody().isPresent() && md.getBody().get().getStatements().stream()
                    .filter(stmt -> stmt instanceof ReturnStmt)
                    .map(stmt -> ((ReturnStmt) stmt).getExpression().orElse(null))
                    .anyMatch(expr -> expr instanceof NullLiteralExpr || 
                            (expr instanceof ConditionalExpr && 
                             (((ConditionalExpr) expr).getThenExpr() instanceof NullLiteralExpr || 
                              ((ConditionalExpr) expr).getElseExpr() instanceof NullLiteralExpr)));
                
                if (shouldAnnotate) {
                    md.addAnnotation("Nullable");
                }
            }
        }, null);
    }

    private static class MethodVisitor extends VoidVisitorAdapter<Void> {

    @Override
    public void visit(MethodDeclaration n, Void arg) {
        processParameters(n);
        super.visit(n, arg);
    }

    @Override
    public void visit(ConstructorDeclaration n, Void arg) {
        processParameters(n);
        super.visit(n, arg);
    }

    private void processParameters(CallableDeclaration<?> n) {
    for (Parameter parameter : n.getParameters()) {
        // Check if the parameter has both @Nullable and @NonNull annotations
        boolean hasNullable = parameter.getAnnotationByName("Nullable").isPresent();
        boolean hasNonNull = parameter.getAnnotationByName("NonNull").isPresent();

        if (hasNullable && hasNonNull) {
            parameter.getAnnotationByName("Nullable").get().remove();
        }

        // Existing code for checking assignments (you can keep this if needed)
        List<AssignExpr> assignments = parameter.findAll(AssignExpr.class, assign -> 
            assign.getTarget().toString().equals(parameter.getNameAsString()));

        for (AssignExpr assignment : assignments) {
            VariableDeclarator var = assignment.findAncestor(VariableDeclarator.class).orElse(null);
            if (var != null && var.getNameAsString().equals(parameter.getNameAsString())) {
                Node parentNode = var.getParentNode().orElse(null);
                List<AnnotationExpr> annotations = null;
                if (parentNode instanceof FieldDeclaration) {
                    annotations = ((FieldDeclaration) parentNode).getAnnotations();
                } else if (parentNode instanceof VariableDeclarationExpr) {
                    annotations = ((VariableDeclarationExpr) parentNode).getAnnotations();
                }
                if (annotations != null && annotations.stream().anyMatch(ann -> ann.getNameAsString().equals("Nullable"))) {
                    parameter.addAnnotation("Nullable");
                    break;
                }
            }
        }
    }
}

}

    public static void removeNullableFromNewConstructors(CompilationUnit cu) {
        for (MethodDeclaration method : cu.findAll(MethodDeclaration.class)) {
            if (method.getAnnotationByName("Nullable").isPresent()) {
                boolean hasNewKeyword =
                        method.findAll(ReturnStmt.class).stream()
                                .anyMatch(
                                        returnStmt ->
                                                returnStmt.getExpression().isPresent()
                                                        && returnStmt.getExpression().get()
                                                                instanceof ObjectCreationExpr);

                if (hasNewKeyword) {
                    method.getAnnotationByName("Nullable").get().remove();
                }
            }
        }
    }

    public static void removeNullableFromConstantMethods(CompilationUnit cu) {
        List<AnnotationExpr> annotationsToRemove = new ArrayList<>();

        new VoidVisitorAdapter<Void>() {
            @Override
            public void visit(MethodDeclaration md, Void arg) {
                super.visit(md, arg);

                // Check if method has a @Nullable annotation
                boolean hasNullableAnnotation = md.getAnnotationByName("Nullable").isPresent();

                // Check if method returns a constant expression
                boolean returnsConstant = md.getBody().isPresent() &&
                                          md.getBody().get().getStatements().size() == 1 &&
                                          md.getBody().get().getStatement(0).isReturnStmt() &&
                                          md.getBody().get().getStatement(0).asReturnStmt().getExpression().isPresent() &&
                                          md.getBody().get().getStatement(0).asReturnStmt().getExpression().get() instanceof LiteralExpr;

                // If method has @Nullable annotation and returns a constant expression, mark the annotation for removal
                if (hasNullableAnnotation && returnsConstant) {
                    md.getAnnotationByName("Nullable").ifPresent(annotationsToRemove::add);
                }
            }
        }.visit(cu, null);

        // Remove all the marked annotations
        annotationsToRemove.forEach(AnnotationExpr::remove);
    }

}
