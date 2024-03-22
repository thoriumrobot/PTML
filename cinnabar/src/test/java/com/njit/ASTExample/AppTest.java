package com.njit.ASTExample;

import static org.junit.Assert.*;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import java.io.File;
import java.io.IOException;
import java.util.*;
import org.junit.Test;

public class AppTest {
    @Test
    public void testTheTest() {
        String code = "class Test { public static void main() {a=b+c; c=d+e; d=null;} }";
/*
        JavaParser parser = new JavaParser();
        ParseResult<CompilationUnit> parseResult = parser.parse(code);
        CompilationUnit compilationUnit = parseResult.getResult().orElse(null);

        BaseNames findnames = new BaseNames();
        findnames.convert(compilationUnit);

        ExpandNames grownames = new ExpandNames(findnames.nameList);
        grownames.convert(compilationUnit);

        while (!grownames.nameList.equals(grownames.nameList_old)) {
            grownames = new ExpandNames(grownames.nameList);
            grownames.convert(compilationUnit);
        }
        ;

        System.out.println("Entries in findnames = " + findnames.nameList.size());
        System.out.println("Entries in grownames = " + grownames.nameList.size());

        assertTrue(findnames.nameList.size() <= grownames.nameList.size());
*/
    }

    @Test
    public void testNameExpansion() {
        // source code directory
        String sourceCodeDirectory = "/home/k/ks225/NullGTN/Sourcegraph/dataset";
/*
        App app = new App();

        try {
            // list of input classes
            List<File> javaFiles = app.getJavaFiles(sourceCodeDirectory);

            // loop over input classes
            for (File file : javaFiles) {
                // parse source code with JavaParser
                CompilationUnit compilationUnit = app.parseJavaFile(file);

                // if parse succeeds...
                if (compilationUnit != null) {
                    // find names under possibly nullable nodes and statements containing
                    // NullLiteralExpr
                    BaseNames findnames = new BaseNames();
                    findnames.convert(compilationUnit);

                    // set the range of AST nodes if desired for experimentation purposes
                    if (findnames.totCount > 200 && findnames.totCount <= 400) {
                        // find names in other statements containing those names until fixpoint is
                        // reached
                        ExpandNames grownames = new ExpandNames(findnames.nameList);
                        grownames.convert(compilationUnit);

                        while (!grownames.nameList.equals(grownames.nameList_old)) {
                            grownames = new ExpandNames(grownames.nameList);
                            grownames.convert(compilationUnit);
                        }
                        ;

                        System.out.println(file.toString());
                        System.out.println("Entries in findnames = " + findnames.nameList.size());
                        System.out.println("Entries in grownames = " + grownames.nameList.size());

                        assertTrue(findnames.nameList.size() <= grownames.nameList.size());
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
*/
    }
}
