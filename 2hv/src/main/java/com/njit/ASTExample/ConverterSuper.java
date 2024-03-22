package com.njit.ASTExample;

import com.github.javaparser.ast.*;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.comments.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.*;
import java.util.*;

// superclass for the methods that traverse, extract information from and transform the ASTs
public class ConverterSuper {
    protected Map<String, Set<Integer>> nameList;
    public int totCount = 0;
    private static final Class[] NODE_TYPES_ARRAY = {
        MethodDeclaration.class,
        Parameter.class,
        FieldDeclaration.class,
        ArrayType.class,
        ClassOrInterfaceType.class,
        VariableDeclarationExpr.class
    };
    private static final Class[] STMT_NODE_TYPES = {
        SynchronizedStmt.class,
        ContinueStmt.class,
        ForStmt.class,
        ForEachStmt.class,
        ExplicitConstructorInvocationStmt.class,
        AssertStmt.class,
        TryStmt.class,
        ThrowStmt.class,
        ReturnStmt.class,
        BlockStmt.class,
        IfStmt.class,
        LocalClassDeclarationStmt.class,
        EmptyStmt.class,
        BreakStmt.class,
        DoStmt.class,
        ExpressionStmt.class,
        LabeledStmt.class,
        WhileStmt.class,
        SwitchStmt.class,
        ArrayAccessExpr.class,
        ArrayCreationExpr.class,
        ArrayInitializerExpr.class,
        AssignExpr.class,
        BinaryExpr.class,
        BooleanLiteralExpr.class,
        CastExpr.class,
        ClassExpr.class,
        ConditionalExpr.class,
        EnclosedExpr.class,
        FieldAccessExpr.class,
        InstanceOfExpr.class,
        LambdaExpr.class,
        MethodCallExpr.class,
        MethodReferenceExpr.class,
        NameExpr.class,
        NormalAnnotationExpr.class,
        ObjectCreationExpr.class,
        SingleMemberAnnotationExpr.class,
        StringLiteralExpr.class,
        SuperExpr.class,
        ThisExpr.class,
        TypeExpr.class,
        UnaryExpr.class,
        VariableDeclarationExpr.class
    };
    private static final Class[] CHOSEN_BY_ABLATION = {
        AnnotationDeclaration.class,
        ArrayAccessExpr.class,
        ArrayCreationExpr.class,
        ArrayCreationLevel.class,
        ArrayInitializerExpr.class,
        ArrayType.class,
        AssertStmt.class,
        AssignExpr.class,
        BinaryExpr.class,
        BlockComment.class,
        BlockStmt.class,
        BreakStmt.class,
        CatchClause.class,
        CharLiteralExpr.class,
        ClassExpr.class,
        ClassOrInterfaceDeclaration.class,
        ConditionalExpr.class,
        ContinueStmt.class,
        DoStmt.class,
        EnclosedExpr.class,
        ExplicitConstructorInvocationStmt.class,
        FieldAccessExpr.class,
        ForEachStmt.class,
        ForStmt.class,
        IfStmt.class,
        ImportDeclaration.class,
        InstanceOfExpr.class,
        IntegerLiteralExpr.class,
        IntersectionType.class,
        LabeledStmt.class,
        LambdaExpr.class,
        LineComment.class,
        LongLiteralExpr.class,
        MemberValuePair.class,
        MethodCallExpr.class,
        MethodDeclaration.class,
        MethodReferenceExpr.class,
        Modifier.class,
        Name.class,
        ObjectCreationExpr.class,
        Parameter.class,
        ReturnStmt.class,
        SwitchEntry.class,
        ThisExpr.class,
        ThrowStmt.class,
        TryStmt.class,
        TypeExpr.class,
        TypeParameter.class,
        UnaryExpr.class,
        UnionType.class,
        VarType.class,
        VariableDeclarator.class,
        WhileStmt.class,
        WildcardType.class
    };

    // return true if the node is an instance of one of the classes in the array
    public static boolean instanceInArray(Class[] A, Node node) {
        for (Class C : A) {
            if (C.isInstance(node)) {
                return true;
            }
        }

        return false;
    }

    // specialized methods for the private arrays
    public static boolean instanceInNODE(Node node) {
        return instanceInArray(NODE_TYPES_ARRAY, node);
    }

    public static boolean instanceInSTMT(Node node) {
        return instanceInArray(STMT_NODE_TYPES, node);
    }

    public static boolean instanceInCHOSEN(Node node) {
        // return instanceInArray(CHOSEN_BY_ABLATION, node);
        return !(node instanceof CatchClause);
        // return true;
    }
}
