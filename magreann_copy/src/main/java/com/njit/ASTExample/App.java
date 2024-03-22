package com.njit.ASTExample;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.*;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.comments.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.json.JSONArray;

public class App {

    /*
    Loop over the input classes, parse AST, convert to graph and save to file.
    main() also tries to prune each AST based on several considerations.
    */
    public static void main(String[] args) {

        File rootDir = new File("/home/ubuntu/minstripped/RIBs");
        processJavaFiles(rootDir, "");
    }

    public static void processJavaFiles(File rootDir, String subdir) {
        String modDir = "/home/ubuntu/llm/magreann_copy/";
        try {

            File[] files = rootDir.listFiles();
            if (files == null) return;

            for (File file : files) {
                if (file.isDirectory()) {
                    processJavaFiles(file, subdir + "/" + file.getName());
                } else if (file.getName().endsWith(".java")
                        && file.toString().contains("/src/main/")) {
                    // predict the nodes
                    ProcessBuilder processBuilder =
                            new ProcessBuilder(
                                    "python",
                                    modDir + "predict_mag.py",
                                    file.toString(),
                                    modDir + "temp.java");
                    Process process = processBuilder.start();

                    // Handle the standard output stream
                    Thread outputThread =
                            new Thread(
                                    () -> {
                                        try (BufferedReader reader =
                                                new BufferedReader(
                                                        new InputStreamReader(
                                                                process.getInputStream()))) {
                                            String line;
                                            while ((line = reader.readLine()) != null) {
                                                System.err.println(line);
                                            }
                                        } catch (IOException e) {
                                            e.printStackTrace();
                                        }
                                    });

                    // Handle the error stream
                    Thread errorThread =
                            new Thread(
                                    () -> {
                                        try (BufferedReader reader =
                                                new BufferedReader(
                                                        new InputStreamReader(
                                                                process.getErrorStream()))) {
                                            String line;
                                            while ((line = reader.readLine()) != null) {
                                                System.err.println("ERROR > " + line);
                                            }
                                        } catch (IOException e) {
                                            e.printStackTrace();
                                        }
                                    });

                    // Start the threads
                    outputThread.start();
                    errorThread.start();

                    // Wait for the process to complete
                    int exitVal = process.waitFor();

                    // Ensure threads have finished
                    outputThread.join();
                    errorThread.join();

                    if (exitVal == 0) {
                        CompilationUnit from = parseJavaFile(new File(modDir + "temp.java"));
                        CompilationUnit cu = parseJavaFile(file);

                        if (from != null) {
                            for (FieldDeclaration fieldFrom :
                                    from.findAll(FieldDeclaration.class)) {
                                Optional<AnnotationExpr> annotation =
                                        fieldFrom.getAnnotationByName("Nullable");
                                if (annotation.isPresent()) {
                                    String fieldNameFrom =
                                            fieldFrom.getVariable(0).getNameAsString();

                                    for (FieldDeclaration fieldTo :
                                            cu.findAll(FieldDeclaration.class)) {
                                        String fieldNameTo =
                                                fieldTo.getVariable(0).getNameAsString();

                                        if (fieldNameFrom.equals(fieldNameTo)) {
                                            fieldTo.addAnnotation(annotation.get().clone());
                                        }
                                    }
                                }
                            }
                        }

                        // save the file
                        String dirpath = "/home/ubuntu/llm/reannotated_mag_copy/RIBs" + subdir + "/";
                        File directory = new File(dirpath);
                        if (!directory.exists()) {
                            directory.mkdirs();
                        }
                        String filePath = dirpath + file.getName();

                        Files.write(
                                Paths.get(filePath),
                                cu.toString().getBytes(StandardCharsets.UTF_8));
                    }

                    File delete_file = new File(modDir + "temp.java");
                    delete_file.delete();
                }
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static List<File> getJavaFiles(String directoryPath) throws IOException {
        try (Stream<Path> paths = Files.walk(Paths.get(directoryPath))) {
            return paths.filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .map(Path::toFile)
                    .collect(Collectors.toList());
        }
    }

    public static CompilationUnit parseJavaFile(File file) {
        JavaParser parser = new JavaParser();
        try {
            ParseResult<CompilationUnit> parseResult = parser.parse(file);
            if (parseResult.isSuccessful()) {
                return parseResult.getResult().orElse(null);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private static void saveJsonToFile(String filePath, JSONArray jsonArray) {
        try (FileWriter fileWriter = new FileWriter(filePath)) {
            fileWriter.write(jsonArray.toString(4));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
