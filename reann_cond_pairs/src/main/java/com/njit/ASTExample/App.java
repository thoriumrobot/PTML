package com.njit.ASTExample;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.*;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.comments.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.nodeTypes.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.*;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import org.json.JSONArray;
import org.json.JSONObject;

public class App {
    static String modDir = "/home/k/ks225/nullproj/excprm/reann_cond_pairs/";
    static File rootDir = new File("/home/k/ks225/nullproj/excprm/minstripped/");
    static String saveDir = "/home/k/ks225/nullproj/excprm/50_20/";

    /*
    Loop over the input classes, parse AST, convert to graph and save to file.
    main() also tries to prune each AST based on several considerations.
    */
    public static void main(String[] args) {
        processJavaFiles(rootDir, "");
    }

    public static void processJavaFiles(File rootDir, String subdir) {
        try {
            File[] files = rootDir.listFiles();
            if (files == null) return;

            for (File file : files) {
                if (file.isDirectory() && file.toString().contains("/src/main")) {
                    HashMap<String, Integer> fileCount = new HashMap<>();
                    HashMap<String, ASTToGraphConverter> converters = new HashMap<>();
                    HashMap<String, HashMap<Integer, Double>> scores = new HashMap<>();

                    List<File[]> javaFilePairs = getJavaFilePairs(file.toString());

                    for (File[] pair : javaFilePairs) {
                        CompilationUnit[] compilationUnits = new CompilationUnit[2];

                        for (int i = 0; i < 2; i++) {
                            if (converters.containsKey(pair[i].getAbsolutePath())) {
                                continue;
                            }

                            compilationUnits[i] = parseJavaFile(pair[i]);
                        }

                        BaseNames[] findnames = new BaseNames[2];
                        ExpandNames[] grownames = new ExpandNames[2];

                        if (compilationUnits[0] != null && compilationUnits[1] != null) {
                            int totCount = 0;

                            for (int i = 0; i < 2; i++) {
                                if (converters.containsKey(pair[i].getAbsolutePath())) {
                                    totCount += converters.get(pair[i].getAbsolutePath()).totCount;
                                    totCount +=
                                            converters
                                                    .get(pair[i].getAbsolutePath())
                                                    .nameList_old
                                                    .size();
                                    continue;
                                }

                                findnames[i] = new BaseNames();
                                findnames[i].convert(compilationUnits[i]);

                                totCount += findnames[i].totCount;
                                totCount += findnames[i].nameList.size();
                            }

                            if (totCount <= 8000) {
                                BufferedWriter writer =
                                        new BufferedWriter(
                                                new FileWriter(modDir + "temp_output.json", true));

                                for (int i = 0; i < 2; i++) {
                                    if (converters.containsKey(pair[i].getAbsolutePath())) {
                                        continue;
                                    }

                                    grownames[i] = new ExpandNames(findnames[i].nameList);
                                    grownames[i].convert(compilationUnits[i]);

                                    while (!grownames[i].nameList.equals(
                                            grownames[i].nameList_old)) {
                                        grownames[i] = new ExpandNames(grownames[i].nameList);
                                        grownames[i].convert(compilationUnits[i]);
                                    }
                                    converters.put(
                                            pair[i].getAbsolutePath(),
                                            new ASTToGraphConverter(grownames[i].nameList));
                                    converters
                                            .get(pair[i].getAbsolutePath())
                                            .convert(compilationUnits[i]);
                                }

                                for (int i = 0; i < 2; i++) {
                                    JSONObject graphJson =
                                            converters.get(pair[i].getAbsolutePath()).toJson();
                                    writer.write(graphJson.toString(4) + "\n");
                                }

                                writer.close();

                                // predict the nodes
                                ProcessBuilder processBuilder =
                                        new ProcessBuilder(
                                                "python", modDir + "GTN_comb/predict.py", "data1");
                                Process process = processBuilder.start();
                                BufferedReader reader =
                                        new BufferedReader(
                                                new InputStreamReader(process.getInputStream()));
                                int exitCode = process.waitFor();

                                if (exitCode == 0) {
                                    fileCount.putIfAbsent(pair[0].getAbsolutePath(), 0);
                                    fileCount.put(
                                            pair[0].getAbsolutePath(),
                                            fileCount.get(pair[0].getAbsolutePath()) + 1);

                                    fileCount.putIfAbsent(pair[1].getAbsolutePath(), 0);
                                    fileCount.put(
                                            pair[1].getAbsolutePath(),
                                            fileCount.get(pair[1].getAbsolutePath()) + 1);

                                    scores.putIfAbsent(pair[0].getAbsolutePath(), new HashMap<>());
                                    scores.putIfAbsent(pair[1].getAbsolutePath(), new HashMap<>());

                                    // Add score to the node's inner HashMap
                                    String line = "";
                                    int lineCount = 0;

                                    while ((line = reader.readLine()) != null) {
                                        int rnno, fidx;
                                        if (lineCount
                                                < converters.get(pair[0].getAbsolutePath())
                                                        .rlvCount) {
                                            fidx = 0;
                                            rnno = lineCount;
                                        } else {
                                            fidx = 1;
                                            rnno =
                                                    lineCount
                                                            - converters.get(
                                                                            pair[0]
                                                                                    .getAbsolutePath())
                                                                    .rlvCount;
                                        }

                                        scores.get(pair[fidx].getAbsolutePath())
                                                .putIfAbsent(rnno, 0.0);
                                        scores.get(pair[fidx].getAbsolutePath())
                                                .put(
                                                        rnno,
                                                        scores.get(pair[fidx].getAbsolutePath())
                                                                        .get(rnno)
                                                                + Double.parseDouble(line));

                                        lineCount++;
                                    }
                                }
                            }
                        }

                        File delete_file = new File(modDir + "temp_output.json");
                        delete_file.delete();
                    }

                    // reannotate all the files
                    for (Map.Entry<String, Integer> entry : fileCount.entrySet()) {
                        File newFile = new File(entry.getKey());
                        ASTToGraphConverter fileConverter = converters.get(entry.getKey());
                        CompilationUnit fileRoot = (CompilationUnit) fileConverter.storedRoot;
                        
                        HashMap<Integer, Double> fileScores = scores.get(entry.getKey());

                        for (Map.Entry<Integer, Double> nodeEntry : fileScores.entrySet()) {
                        Node node;
                        if (!fileConverter.rlvNodes.isEmpty() && nodeEntry.getKey() < fileConverter.rlvNodes.size()) {
                            node = fileConverter.rlvNodes.get(nodeEntry.getKey());
                        } else {
                        continue;
                        }

                            if (((node instanceof MethodDeclaration)
                                            && nodeEntry.getValue()
                                                    > fileCount.get(entry.getKey()) * 0.7341)
                                    || ((node instanceof FieldDeclaration)
                                            && nodeEntry.getValue()
                                                    > fileCount.get(entry.getKey()) * 0.8449)) {
                                ((NodeWithAnnotations<?>) node).addAnnotation("Nullable");
                            }
                        }

                        // save the file
                        Path rootPath = rootDir.toPath();
                        Path rootSub = rootPath.relativize(newFile.toPath());

                        String dirPath = saveDir + subdir + "/" + rootSub.getParent() + "/";
                        File directory = new File(dirPath);

                        if (!directory.exists()) {
                            directory.mkdirs();
                        }

                        String filePath = dirPath + newFile.getName();
                        Files.write(
                                Paths.get(filePath),
                                fileRoot.toString().getBytes(StandardCharsets.UTF_8));
                    }
                        
                        //annotate Parameters
                        NullableParameterModifier.processProject(new File(saveDir + subdir + "/"));
                        
                        //annotate from Parameters
                        NullableProcessorByName.nPBM(saveDir + subdir + "/");
                } else if (file.isDirectory()) {
                    processJavaFiles(file, subdir + "/" + file.getName());
                }
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    private static List<File> getJavaFiles(File dir) {
        List<File> javaFiles = new ArrayList<>();

        File[] files = dir.listFiles();
        if (files == null) {
            return javaFiles;
        }

        for (File file : files) {
            if (file.isDirectory()) {
                javaFiles.addAll(getJavaFiles(file));
            } else if (file.getName().endsWith(".java")) {
                javaFiles.add(file);
            }
        }

        return javaFiles;
    }

    public static List<File[]> getJavaFilePairs(String directoryPath) {
        List<File[]> pairs = new ArrayList<>();
        List<File> javaFiles = getJavaFiles(new File(directoryPath));

        for (int i = 0; i < javaFiles.size(); i++) {
            for (int j = i + 1; j < javaFiles.size(); j++) {
                pairs.add(new File[] {javaFiles.get(i), javaFiles.get(j)});
            }
        }

        return pairs;
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
