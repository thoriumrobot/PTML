package com.njit.ASTExample;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.json.JSONObject;

public class App {
    public static void main(String[] args) {
        Path startPath = Paths.get(System.getProperty("dir"));
        String outputJsonFile = "output_prune.json";

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputJsonFile, true));

            Deque<Path> dirs = new ArrayDeque<>();
            dirs.add(startPath);

            while (!dirs.isEmpty()) {
                Path dir = dirs.remove();

                try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
                    for (Path path : stream) {
                        if (Files.isDirectory(path)) {
                            dirs.add(path);
                        } else if (path.toString().endsWith(".java")) {
                            File file = path.toFile();
                            CompilationUnit compilationUnit = parseJavaFile(file);

                            if (compilationUnit != null) {
                                BaseNames findnames = new BaseNames();
                                findnames.convert(compilationUnit);

                                if (findnames.totCount >= 501 && findnames.totCount < 1200) {
                                    ExpandNames grownames = new ExpandNames(findnames.nameList);
                                    grownames.convert(compilationUnit);

                                    while (!grownames.nameList.equals(grownames.nameList_old)) {
                                        grownames = new ExpandNames(grownames.nameList);
                                        grownames.convert(compilationUnit);
                                    }

                                    ASTToGraphConverter converter =
                                            new ASTToGraphConverter(grownames.nameList);
                                    converter.convert(compilationUnit);

                                    if (converter.foundAnnotation) {
                                        System.out.println(file.toString());

                                        JSONObject graphJson = converter.toJson();
                                        writer.write(graphJson.toString(4) + "\n");
                                        writer.flush();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            writer.close();
        } catch (IOException e) {
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
}
