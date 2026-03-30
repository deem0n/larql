/// LQL Abstract Syntax Tree.
///
/// Every LQL statement parses into one `Statement` variant.
/// The executor dispatches each variant to the appropriate backend.

#[derive(Debug, Clone)]
pub enum Statement {
    // ── Lifecycle ──
    Extract {
        model: String,
        output: String,
        components: Option<Vec<Component>>,
        layers: Option<Range>,
    },
    Compile {
        vindex: VindexRef,
        output: String,
        format: Option<OutputFormat>,
    },
    Diff {
        a: VindexRef,
        b: VindexRef,
        layer: Option<u32>,
        relation: Option<String>,
        limit: Option<u32>,
    },
    Use {
        target: UseTarget,
    },

    // ── Query ──
    Walk {
        prompt: String,
        top: Option<u32>,
        layers: Option<Range>,
        mode: Option<WalkMode>,
        compare: bool,
    },
    /// Full inference with attention — requires model weights.
    Infer {
        prompt: String,
        top: Option<u32>,
        compare: bool,
    },
    Select {
        fields: Vec<Field>,
        conditions: Vec<Condition>,
        nearest: Option<NearestClause>,
        order: Option<OrderBy>,
        limit: Option<u32>,
    },
    Describe {
        entity: String,
        layer: Option<u32>,
        relations_only: bool,
    },
    Explain {
        prompt: String,
        layers: Option<Range>,
        verbose: bool,
    },

    // ── Mutation ──
    Insert {
        entity: String,
        relation: String,
        target: String,
        layer: Option<u32>,
        confidence: Option<f32>,
    },
    Delete {
        conditions: Vec<Condition>,
    },
    Update {
        set: Vec<Assignment>,
        conditions: Vec<Condition>,
    },
    Merge {
        source: String,
        target: Option<String>,
        conflict: Option<ConflictStrategy>,
    },

    // ── Introspection ──
    ShowRelations {
        layer: Option<u32>,
        with_examples: bool,
    },
    ShowLayers {
        range: Option<Range>,
    },
    ShowFeatures {
        layer: u32,
        conditions: Vec<Condition>,
        limit: Option<u32>,
    },
    ShowModels,
    Stats {
        vindex: Option<String>,
    },

    // ── Pipe ──
    Pipe {
        left: Box<Statement>,
        right: Box<Statement>,
    },
}

#[derive(Debug, Clone)]
pub enum VindexRef {
    Path(String),
    Current,
}

#[derive(Debug, Clone)]
pub enum UseTarget {
    Vindex(String),
    Model { id: String, auto_extract: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkMode {
    Hybrid,
    Pure,
    Dense,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Safetensors,
    Gguf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictStrategy {
    KeepSource,
    KeepTarget,
    HighestConfidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    FfnGate,
    FfnDown,
    FfnUp,
    Embeddings,
    AttnOv,
    AttnQk,
}

#[derive(Debug, Clone)]
pub struct Range {
    pub start: u32,
    pub end: u32,
}

#[derive(Debug, Clone)]
pub enum Field {
    Star,
    Named(String),
}

#[derive(Debug, Clone)]
pub struct Condition {
    pub field: String,
    pub op: CompareOp,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub enum CompareOp {
    Eq,
    Neq,
    Gt,
    Lt,
    Gte,
    Lte,
    Like,
    In,
}

#[derive(Debug, Clone)]
pub enum Value {
    String(String),
    Number(f64),
    Integer(i64),
    List(Vec<Value>),
}

#[derive(Debug, Clone)]
pub struct NearestClause {
    pub entity: String,
    pub layer: u32,
}

#[derive(Debug, Clone)]
pub struct OrderBy {
    pub field: String,
    pub descending: bool,
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub field: String,
    pub value: Value,
}
