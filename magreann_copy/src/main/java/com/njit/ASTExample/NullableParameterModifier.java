package com.njit.ASTExample;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashSet;
import java.util.Set;

public class NullableParameterModifier {

    private static Set<String> methodsToAnnotate = new HashSet<>();

    public static void processProject(File projectDir) {
        for (File file : projectDir.listFiles()) {
            if (file.isDirectory()) {
                processProject(file);
            } else if (file.getName().endsWith(".java")) {
                try {
                    JavaParser parser = new JavaParser();
                    ParseResult<CompilationUnit> result = parser.parse(file);
                    if (result.isSuccessful() && result.getResult().isPresent()) {
                        CompilationUnit cu = result.getResult().get();
                        annotateNullableParameters(cu);
                        // Optionally, write the modified CU back to file:
                        Files.write(file.toPath(), cu.toString().getBytes());
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private static void annotateNullableParameters(CompilationUnit cu) {
        // First pass: find all method calls
        // (Note: This does not check for @Nullable arguments because Expressions don't have
        // annotations)
        cu.accept(
                new VoidVisitorAdapter<Void>() {
                    @Override
                    public void visit(MethodCallExpr n, Void arg) {
                        methodsToAnnotate.add(n.getNameAsString());
                        super.visit(n, arg);
                    }
                },
                null);

        // Second pass: annotate the parameters of the methods found in the first pass
        cu.accept(
                new VoidVisitorAdapter<Void>() {
                    @Override
                    public void visit(MethodDeclaration n, Void arg) {
                        if (methodsToAnnotate.contains(n.getNameAsString())) {
                            n.getParameters()
                                    .forEach(
                                            param -> {
                                                if (!param.getAnnotations().stream()
                                                        .anyMatch(
                                                                anno ->
                                                                        anno.getNameAsString()
                                                                                .equals(
                                                                                        "Nullable"))) {
                                                    param.addAnnotation("Nullable");
                                                }
                                            });
                        }
                        super.visit(n, arg);
                    }
                },
                null);
    }
}
