//! LQL Parser Demo — parse every statement type from the spec and display the AST.
//!
//! Run: cargo run -p larql-lql --example parser_demo

use larql_lql::parse;

fn main() {
    println!("=== LQL Parser Demo ===\n");

    // ── Lifecycle Statements ──
    section("Lifecycle");

    demo(
        "EXTRACT (minimal)",
        r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";"#,
    );

    demo(
        "EXTRACT (full)",
        r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" COMPONENTS FFN_GATE, FFN_DOWN, FFN_UP, EMBEDDINGS LAYERS 0-33;"#,
    );

    demo(
        "COMPILE (safetensors)",
        r#"COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;"#,
    );

    demo(
        "COMPILE (gguf, from path)",
        r#"COMPILE "gemma3.vindex" INTO MODEL "out/" FORMAT gguf;"#,
    );

    demo(
        "DIFF (path vs CURRENT)",
        r#"DIFF "gemma3-4b.vindex" CURRENT;"#,
    );

    demo(
        "DIFF (with LIMIT)",
        r#"DIFF "a.vindex" "b.vindex" LIMIT 20;"#,
    );

    demo(
        "USE (vindex)",
        r#"USE "gemma3-4b.vindex";"#,
    );

    demo(
        "USE MODEL",
        r#"USE MODEL "google/gemma-3-4b-it";"#,
    );

    demo(
        "USE MODEL AUTO_EXTRACT",
        r#"USE MODEL "google/gemma-3-4b-it" AUTO_EXTRACT;"#,
    );

    // ── Query Statements ──
    section("Query");

    demo(
        "WALK (minimal)",
        r#"WALK "The capital of France is";"#,
    );

    demo(
        "WALK (full options)",
        r#"WALK "The capital of France is" TOP 5 LAYERS 25-33 MODE hybrid COMPARE;"#,
    );

    demo(
        "WALK (all modes)",
        r#"WALK "test" MODE pure;"#,
    );

    demo(
        "SELECT (star, no WHERE)",
        "SELECT * FROM EDGES LIMIT 5;",
    );

    demo(
        "SELECT (fields + WHERE + ORDER + LIMIT)",
        r#"SELECT entity, relation, target, confidence FROM EDGES WHERE entity = "France" ORDER BY confidence DESC LIMIT 10;"#,
    );

    demo(
        "SELECT (multiple conditions)",
        r#"SELECT * FROM EDGES WHERE relation = "capital-of" AND confidence > 0.5;"#,
    );

    demo(
        "SELECT (by layer + feature)",
        "SELECT * FROM EDGES WHERE layer = 26 AND feature = 9515;",
    );

    demo(
        "SELECT (NEAREST TO)",
        r#"SELECT entity, target, distance FROM EDGES NEAREST TO "Mozart" AT LAYER 26 LIMIT 20;"#,
    );

    demo(
        "SELECT (LIKE)",
        r#"SELECT * FROM EDGES WHERE entity LIKE "Fran%";"#,
    );

    demo(
        "SELECT (IN list)",
        r#"SELECT * FROM EDGES WHERE entity IN ("France", "Germany", "Japan");"#,
    );

    demo(
        "DESCRIBE (minimal)",
        r#"DESCRIBE "France";"#,
    );

    demo(
        "DESCRIBE (AT LAYER)",
        r#"DESCRIBE "Mozart" AT LAYER 26;"#,
    );

    demo(
        "DESCRIBE (RELATIONS ONLY)",
        r#"DESCRIBE "France" RELATIONS ONLY;"#,
    );

    demo(
        "EXPLAIN WALK",
        r#"EXPLAIN WALK "The capital of France is";"#,
    );

    demo(
        "EXPLAIN WALK (with options)",
        r#"EXPLAIN WALK "prompt" LAYERS 24-33 VERBOSE;"#,
    );

    // ── Mutation Statements ──
    section("Mutation");

    demo(
        "INSERT (minimal)",
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John Coyle", "lives-in", "Colchester");"#,
    );

    demo(
        "INSERT (with layer + confidence)",
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John", "occupation", "engineer") AT LAYER 26 CONFIDENCE 0.8;"#,
    );

    demo(
        "DELETE (single condition)",
        r#"DELETE FROM EDGES WHERE entity = "outdated";"#,
    );

    demo(
        "DELETE (AND conditions)",
        r#"DELETE FROM EDGES WHERE entity = "John Coyle" AND relation = "lives-in";"#,
    );

    demo(
        "UPDATE",
        r#"UPDATE EDGES SET target = "London" WHERE entity = "John Coyle" AND relation = "lives-in";"#,
    );

    demo(
        "MERGE (minimal)",
        r#"MERGE "medical-knowledge.vindex";"#,
    );

    demo(
        "MERGE (full)",
        r#"MERGE "medical-knowledge.vindex" INTO "gemma3-4b.vindex" ON CONFLICT HIGHEST_CONFIDENCE;"#,
    );

    // ── Introspection Statements ──
    section("Introspection");

    demo("SHOW RELATIONS", "SHOW RELATIONS;");
    demo("SHOW RELATIONS WITH EXAMPLES", "SHOW RELATIONS WITH EXAMPLES;");
    demo("SHOW RELATIONS AT LAYER", "SHOW RELATIONS AT LAYER 26;");
    demo("SHOW LAYERS", "SHOW LAYERS;");
    demo("SHOW LAYERS (range)", "SHOW LAYERS RANGE 0-10;");
    demo("SHOW FEATURES", "SHOW FEATURES 26;");
    demo(
        "SHOW FEATURES (with filter)",
        r#"SHOW FEATURES 26 WHERE relation = "capital-of" LIMIT 5;"#,
    );
    demo("SHOW MODELS", "SHOW MODELS;");
    demo("STATS", "STATS;");
    demo("STATS (with path)", r#"STATS "gemma3.vindex";"#);

    // ── Pipe Operator ──
    section("Pipe Operator");

    demo(
        "WALK |> EXPLAIN",
        r#"WALK "The capital of France is" TOP 5 |> EXPLAIN WALK "The capital of France is";"#,
    );

    // ── Comments ──
    section("Comments");

    demo(
        "Leading comment",
        "-- This is a comment\nSTATS;",
    );

    demo(
        "Inline comment",
        "STATS; -- trailing comment",
    );

    demo(
        "Multi-line with comments",
        "-- Act 1\nSHOW RELATIONS;\n-- Act 2",
    );

    // ── Demo Script (from spec) ──
    section("Full Demo Script");

    let demo_stmts = vec![
        r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";"#,
        r#"USE "gemma3-4b.vindex";"#,
        "STATS;",
        "SHOW RELATIONS WITH EXAMPLES;",
        r#"DESCRIBE "France";"#,
        r#"SELECT entity, target, confidence FROM EDGES WHERE relation = "capital-of" ORDER BY confidence DESC LIMIT 10;"#,
        r#"WALK "The capital of France is" TOP 5 COMPARE;"#,
        r#"EXPLAIN WALK "The capital of France is";"#,
        r#"WALK "Where does John Coyle live?" TOP 5;"#,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John Coyle", "lives-in", "Colchester");"#,
        r#"DESCRIBE "John Coyle";"#,
        r#"DIFF "gemma3-4b.vindex" CURRENT;"#,
        r#"COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;"#,
    ];

    let mut ok = 0;
    let mut fail = 0;
    for (i, input) in demo_stmts.iter().enumerate() {
        match parse(input) {
            Ok(_) => {
                println!("  {:2}. OK   {}", i + 1, truncate(input, 70));
                ok += 1;
            }
            Err(e) => {
                println!("  {:2}. FAIL {} — {}", i + 1, truncate(input, 50), e);
                fail += 1;
            }
        }
    }
    println!("\n  {ok}/{} statements parsed successfully.", ok + fail);

    println!("\n=== Done ===");
}

fn section(name: &str) {
    println!("\n── {} ──\n", name);
}

fn demo(label: &str, input: &str) {
    match parse(input) {
        Ok(stmt) => {
            println!("  {label}:");
            println!("    Input: {}", truncate(input, 80));
            println!("    AST:   {:?}", stmt);
            println!();
        }
        Err(e) => {
            println!("  {label}: PARSE ERROR");
            println!("    Input: {}", truncate(input, 80));
            println!("    Error: {e}");
            println!();
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s
    }
}
