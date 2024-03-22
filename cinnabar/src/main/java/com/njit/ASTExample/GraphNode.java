package com.njit.ASTExample;

import com.github.javaparser.ast.Node;
import java.util.*;

public class GraphNode {
    // JavaParser node
    public Node node;

    public int id;
    public ArrayList<String> type;
    public int nullable;

    // boolean flag used to select the node based on custom criteria
    public boolean chooseNode;

    public GraphNode(Node node, int id, ArrayList<String> type, int nullable) {
        this.node = node;
        this.id = id;
        this.type = new ArrayList<>();
        this.nullable = nullable;
        chooseNode = false;

        this.type.addAll(type);
    }

    public GraphNode(Node node, int id, String type, int nullable) {
        this.node = node;
        this.id = id;
        this.type = new ArrayList<>();
        this.nullable = nullable;
        chooseNode = false;

        this.type.add(type);
    }

    public GraphNode(Node node) {
        this.node = node;
        this.id = 0;
        this.type = new ArrayList<>();
        this.nullable = 0;
        chooseNode = false;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("GraphNode {");
        sb.append("id=").append(id);
        sb.append(", type='").append(type.toString()).append('\'');
        sb.append(", nullable=").append(nullable);
        sb.append('}');
        return sb.toString();
    }
}
