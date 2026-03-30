/// LQL Parser — recursive descent from token stream to AST.

use crate::ast::*;
use crate::lexer::{Keyword, Token};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

#[derive(Debug, Clone)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parse error: {}", self.0)
    }
}

impl std::error::Error for ParseError {}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    pub fn parse(&mut self) -> Result<Statement, ParseError> {
        let stmt = self.parse_statement()?;
        // Check for pipe operator
        if self.check_pipe() {
            self.advance(); // consume |>
            let right = self.parse_statement()?;
            Ok(Statement::Pipe {
                left: Box::new(stmt),
                right: Box::new(right),
            })
        } else {
            Ok(stmt)
        }
    }

    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        match self.peek() {
            Token::Keyword(Keyword::Extract) => self.parse_extract(),
            Token::Keyword(Keyword::Compile) => self.parse_compile(),
            Token::Keyword(Keyword::Diff) => self.parse_diff(),
            Token::Keyword(Keyword::Use) => self.parse_use(),
            Token::Keyword(Keyword::Walk) => self.parse_walk(),
            Token::Keyword(Keyword::Infer) => self.parse_infer(),
            Token::Keyword(Keyword::Select) => self.parse_select(),
            Token::Keyword(Keyword::Describe) => self.parse_describe(),
            Token::Keyword(Keyword::Explain) => self.parse_explain(),
            Token::Keyword(Keyword::Insert) => self.parse_insert(),
            Token::Keyword(Keyword::Delete) => self.parse_delete(),
            Token::Keyword(Keyword::Update) => self.parse_update(),
            Token::Keyword(Keyword::Merge) => self.parse_merge(),
            Token::Keyword(Keyword::Show) => self.parse_show(),
            Token::Keyword(Keyword::Stats) => self.parse_stats(),
            _ => Err(ParseError(format!(
                "expected statement keyword, got {:?}",
                self.peek()
            ))),
        }
    }

    // ── Lifecycle ──

    fn parse_extract(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Extract)?;
        self.expect_keyword(Keyword::Model)?;
        let model = self.expect_string()?;
        self.expect_keyword(Keyword::Into)?;
        let output = self.expect_string()?;

        let mut components = None;
        let mut layers = None;

        loop {
            match self.peek() {
                Token::Keyword(Keyword::Components) => {
                    self.advance();
                    components = Some(self.parse_component_list()?);
                }
                Token::Keyword(Keyword::Layers) => {
                    self.advance();
                    layers = Some(self.parse_range()?);
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Extract { model, output, components, layers })
    }

    fn parse_compile(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Compile)?;
        let vindex = self.parse_vindex_ref()?;
        self.expect_keyword(Keyword::Into)?;
        self.expect_keyword(Keyword::Model)?;
        let output = self.expect_string()?;

        let mut format = None;
        if self.check_keyword(Keyword::Format) {
            self.advance();
            format = Some(self.parse_output_format()?);
        }

        self.eat_semicolon();
        Ok(Statement::Compile { vindex, output, format })
    }

    fn parse_diff(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Diff)?;
        let a = self.parse_vindex_ref()?;
        let b = self.parse_vindex_ref()?;

        let mut layer = None;
        let mut relation = None;
        let mut limit = None;

        loop {
            match self.peek() {
                Token::Keyword(Keyword::Layer) => {
                    self.advance();
                    layer = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Relations) | Token::Keyword(Keyword::Limit) => {
                    if self.check_keyword(Keyword::Limit) {
                        self.advance();
                        limit = Some(self.expect_u32()?);
                    } else {
                        self.advance();
                        relation = Some(self.expect_string()?);
                    }
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Diff { a, b, layer, relation, limit })
    }

    fn parse_use(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Use)?;

        let target = if self.check_keyword(Keyword::Model) {
            self.advance();
            let id = self.expect_string()?;
            let auto_extract = self.check_keyword(Keyword::AutoExtract);
            if auto_extract {
                self.advance();
            }
            UseTarget::Model { id, auto_extract }
        } else {
            let path = self.expect_string()?;
            UseTarget::Vindex(path)
        };

        self.eat_semicolon();
        Ok(Statement::Use { target })
    }

    // ── Query ──

    fn parse_walk(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Walk)?;
        let prompt = self.expect_string()?;

        let mut top = None;
        let mut layers = None;
        let mut mode = None;
        let mut compare = false;

        loop {
            match self.peek() {
                Token::Keyword(Keyword::Top) => {
                    self.advance();
                    top = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Layers) => {
                    self.advance();
                    if self.check_keyword(Keyword::All) {
                        self.advance();
                        // ALL means no range filter
                    } else {
                        layers = Some(self.parse_range()?);
                    }
                }
                Token::Keyword(Keyword::Mode) => {
                    self.advance();
                    mode = Some(self.parse_walk_mode()?);
                }
                Token::Keyword(Keyword::Compare) => {
                    self.advance();
                    compare = true;
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Walk { prompt, top, layers, mode, compare })
    }

    fn parse_infer(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Infer)?;
        let prompt = self.expect_string()?;

        let mut top = None;
        let mut compare = false;

        loop {
            match self.peek() {
                Token::Keyword(Keyword::Top) => {
                    self.advance();
                    top = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Compare) => {
                    self.advance();
                    compare = true;
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Infer { prompt, top, compare })
    }

    fn parse_select(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Select)?;

        // Fields
        let fields = self.parse_field_list()?;

        // FROM EDGES
        self.expect_keyword(Keyword::From)?;
        self.expect_keyword(Keyword::Edges)?;

        // Optional NEAREST TO
        let mut nearest = None;
        if self.check_keyword(Keyword::Nearest) {
            self.advance();
            self.expect_keyword(Keyword::To)?;
            let entity = self.expect_string()?;
            self.expect_keyword(Keyword::At)?;
            self.expect_keyword(Keyword::Layer)?;
            let layer = self.expect_u32()?;
            nearest = Some(NearestClause { entity, layer });
        }

        // Optional WHERE
        let conditions = if self.check_keyword(Keyword::Where) {
            self.advance();
            self.parse_conditions()?
        } else {
            vec![]
        };

        // Optional ORDER BY
        let order = if self.check_keyword(Keyword::Order) {
            self.advance();
            self.expect_keyword(Keyword::By)?;
            Some(self.parse_order_by()?)
        } else {
            None
        };

        // Optional LIMIT
        let limit = if self.check_keyword(Keyword::Limit) {
            self.advance();
            Some(self.expect_u32()?)
        } else {
            None
        };

        self.eat_semicolon();
        Ok(Statement::Select { fields, conditions, nearest, order, limit })
    }

    fn parse_describe(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Describe)?;
        let entity = self.expect_string()?;

        let mut layer = None;
        let mut relations_only = false;

        loop {
            match self.peek() {
                Token::Keyword(Keyword::At) => {
                    self.advance();
                    self.expect_keyword(Keyword::Layer)?;
                    layer = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Relations) => {
                    self.advance();
                    self.expect_keyword(Keyword::Only)?;
                    relations_only = true;
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Describe { entity, layer, relations_only })
    }

    fn parse_explain(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Explain)?;
        // EXPLAIN WALK <prompt>
        self.expect_keyword(Keyword::Walk)?;
        let prompt = self.expect_string()?;

        let mut layers = None;
        let mut verbose = false;

        loop {
            match self.peek() {
                Token::Keyword(Keyword::Layers) => {
                    self.advance();
                    layers = Some(self.parse_range()?);
                }
                Token::Keyword(Keyword::Verbose) => {
                    self.advance();
                    verbose = true;
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Explain { prompt, layers, verbose })
    }

    // ── Mutation ──

    fn parse_insert(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Insert)?;
        self.expect_keyword(Keyword::Into)?;
        self.expect_keyword(Keyword::Edges)?;

        // (entity, relation, target)
        self.expect_token(&Token::LParen)?;
        self.expect_ident_eq("entity")?;
        self.expect_token(&Token::Comma)?;
        self.expect_ident_eq("relation")?;
        self.expect_token(&Token::Comma)?;
        self.expect_ident_eq("target")?;
        self.expect_token(&Token::RParen)?;

        // VALUES (e, r, t)
        self.expect_keyword(Keyword::Values)?;
        self.expect_token(&Token::LParen)?;
        let entity = self.expect_string()?;
        self.expect_token(&Token::Comma)?;
        let relation = self.expect_string()?;
        self.expect_token(&Token::Comma)?;
        let target = self.expect_string()?;
        self.expect_token(&Token::RParen)?;

        let mut layer = None;
        let mut confidence = None;

        loop {
            match self.peek() {
                Token::Keyword(Keyword::At) => {
                    self.advance();
                    self.expect_keyword(Keyword::Layer)?;
                    layer = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Confidence) => {
                    self.advance();
                    confidence = Some(self.expect_f32()?);
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Insert { entity, relation, target, layer, confidence })
    }

    fn parse_delete(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Delete)?;
        self.expect_keyword(Keyword::From)?;
        self.expect_keyword(Keyword::Edges)?;
        self.expect_keyword(Keyword::Where)?;
        let conditions = self.parse_conditions()?;
        self.eat_semicolon();
        Ok(Statement::Delete { conditions })
    }

    fn parse_update(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Update)?;
        self.expect_keyword(Keyword::Edges)?;
        self.expect_keyword(Keyword::Set)?;

        let set = self.parse_assignments()?;

        self.expect_keyword(Keyword::Where)?;
        let conditions = self.parse_conditions()?;
        self.eat_semicolon();
        Ok(Statement::Update { set, conditions })
    }

    fn parse_merge(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Merge)?;
        let source = self.expect_string()?;

        let mut target = None;
        let mut conflict = None;

        if self.check_keyword(Keyword::Into) {
            self.advance();
            target = Some(self.expect_string()?);
        }

        if self.check_keyword(Keyword::On) {
            self.advance();
            self.expect_keyword(Keyword::Conflict)?;
            conflict = Some(self.parse_conflict_strategy()?);
        }

        self.eat_semicolon();
        Ok(Statement::Merge { source, target, conflict })
    }

    // ── Introspection ──

    fn parse_show(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Show)?;

        match self.peek() {
            Token::Keyword(Keyword::Relations) => {
                self.advance();
                let mut layer = None;
                let mut with_examples = false;

                loop {
                    match self.peek() {
                        Token::Keyword(Keyword::At) => {
                            self.advance();
                            self.expect_keyword(Keyword::Layer)?;
                            layer = Some(self.expect_u32()?);
                        }
                        Token::Keyword(Keyword::With) => {
                            self.advance();
                            self.expect_keyword(Keyword::Examples)?;
                            with_examples = true;
                        }
                        _ => break,
                    }
                }
                self.eat_semicolon();
                Ok(Statement::ShowRelations { layer, with_examples })
            }
            Token::Keyword(Keyword::Layers) => {
                self.advance();
                let range = if self.check_keyword(Keyword::Range) {
                    self.advance();
                    Some(self.parse_range()?)
                } else if matches!(self.peek(), Token::IntegerLit(_)) {
                    Some(self.parse_range()?)
                } else {
                    None
                };
                self.eat_semicolon();
                Ok(Statement::ShowLayers { range })
            }
            Token::Keyword(Keyword::Features) => {
                self.advance();
                let layer = self.expect_u32()?;

                let conditions = if self.check_keyword(Keyword::Where) {
                    self.advance();
                    self.parse_conditions()?
                } else {
                    vec![]
                };

                let limit = if self.check_keyword(Keyword::Limit) {
                    self.advance();
                    Some(self.expect_u32()?)
                } else {
                    None
                };

                self.eat_semicolon();
                Ok(Statement::ShowFeatures { layer, conditions, limit })
            }
            Token::Keyword(Keyword::Models) => {
                self.advance();
                self.eat_semicolon();
                Ok(Statement::ShowModels)
            }
            _ => Err(ParseError(format!(
                "expected RELATIONS, LAYERS, FEATURES, or MODELS after SHOW, got {:?}",
                self.peek()
            ))),
        }
    }

    fn parse_stats(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Stats)?;
        let vindex = if let Token::StringLit(_) = self.peek() {
            Some(self.expect_string()?)
        } else {
            None
        };
        self.eat_semicolon();
        Ok(Statement::Stats { vindex })
    }

    // ── Helpers ──

    fn parse_vindex_ref(&mut self) -> Result<VindexRef, ParseError> {
        if self.check_keyword(Keyword::Current) {
            self.advance();
            Ok(VindexRef::Current)
        } else {
            Ok(VindexRef::Path(self.expect_string()?))
        }
    }

    fn parse_range(&mut self) -> Result<Range, ParseError> {
        let start = self.expect_u32()?;
        self.expect_token(&Token::Dash)?;
        let end = self.expect_u32()?;
        Ok(Range { start, end })
    }

    fn parse_walk_mode(&mut self) -> Result<WalkMode, ParseError> {
        match self.peek() {
            Token::Keyword(Keyword::Hybrid) => { self.advance(); Ok(WalkMode::Hybrid) }
            Token::Keyword(Keyword::Pure) => { self.advance(); Ok(WalkMode::Pure) }
            Token::Keyword(Keyword::Dense) => { self.advance(); Ok(WalkMode::Dense) }
            _ => Err(ParseError(format!("expected HYBRID, PURE, or DENSE, got {:?}", self.peek()))),
        }
    }

    fn parse_output_format(&mut self) -> Result<OutputFormat, ParseError> {
        match self.peek() {
            Token::Keyword(Keyword::Safetensors) => { self.advance(); Ok(OutputFormat::Safetensors) }
            Token::Keyword(Keyword::Gguf) => { self.advance(); Ok(OutputFormat::Gguf) }
            _ => Err(ParseError(format!("expected SAFETENSORS or GGUF, got {:?}", self.peek()))),
        }
    }

    fn parse_conflict_strategy(&mut self) -> Result<ConflictStrategy, ParseError> {
        match self.peek() {
            Token::Keyword(Keyword::KeepSource) => { self.advance(); Ok(ConflictStrategy::KeepSource) }
            Token::Keyword(Keyword::KeepTarget) => { self.advance(); Ok(ConflictStrategy::KeepTarget) }
            Token::Keyword(Keyword::HighestConfidence) => { self.advance(); Ok(ConflictStrategy::HighestConfidence) }
            _ => Err(ParseError(format!("expected KEEP_SOURCE, KEEP_TARGET, or HIGHEST_CONFIDENCE, got {:?}", self.peek()))),
        }
    }

    fn parse_component_list(&mut self) -> Result<Vec<Component>, ParseError> {
        let mut components = vec![self.parse_component()?];
        while self.check_comma() {
            self.advance();
            components.push(self.parse_component()?);
        }
        Ok(components)
    }

    fn parse_component(&mut self) -> Result<Component, ParseError> {
        match self.peek() {
            Token::Keyword(Keyword::FfnGate) => { self.advance(); Ok(Component::FfnGate) }
            Token::Keyword(Keyword::FfnDown) => { self.advance(); Ok(Component::FfnDown) }
            Token::Keyword(Keyword::FfnUp) => { self.advance(); Ok(Component::FfnUp) }
            Token::Keyword(Keyword::Embeddings) => { self.advance(); Ok(Component::Embeddings) }
            Token::Keyword(Keyword::AttnOv) => { self.advance(); Ok(Component::AttnOv) }
            Token::Keyword(Keyword::AttnQk) => { self.advance(); Ok(Component::AttnQk) }
            // Also accept unquoted identifiers for convenience
            Token::Ident(ref s) => {
                let c = match s.to_lowercase().as_str() {
                    "ffn_gate" => Component::FfnGate,
                    "ffn_down" => Component::FfnDown,
                    "ffn_up" => Component::FfnUp,
                    "embeddings" => Component::Embeddings,
                    "attn_ov" => Component::AttnOv,
                    "attn_qk" => Component::AttnQk,
                    _ => return Err(ParseError(format!("unknown component: {s}"))),
                };
                self.advance();
                Ok(c)
            }
            _ => Err(ParseError(format!("expected component name, got {:?}", self.peek()))),
        }
    }

    fn parse_field_list(&mut self) -> Result<Vec<Field>, ParseError> {
        if matches!(self.peek(), Token::Star) {
            self.advance();
            return Ok(vec![Field::Star]);
        }

        let mut fields = vec![self.parse_field()?];
        while self.check_comma() {
            self.advance();
            fields.push(self.parse_field()?);
        }
        Ok(fields)
    }

    fn parse_field(&mut self) -> Result<Field, ParseError> {
        match self.peek() {
            Token::Star => {
                self.advance();
                Ok(Field::Star)
            }
            Token::Ident(ref name) => {
                let name = name.clone();
                self.advance();
                Ok(Field::Named(name))
            }
            // Some field names collide with keywords (e.g. "layer", "confidence")
            Token::Keyword(kw) => {
                let name = format!("{:?}", kw).to_lowercase();
                self.advance();
                Ok(Field::Named(name))
            }
            _ => Err(ParseError(format!("expected field name, got {:?}", self.peek()))),
        }
    }

    fn parse_conditions(&mut self) -> Result<Vec<Condition>, ParseError> {
        let mut conditions = vec![self.parse_condition()?];
        while self.check_keyword(Keyword::And) {
            self.advance();
            conditions.push(self.parse_condition()?);
        }
        Ok(conditions)
    }

    fn parse_condition(&mut self) -> Result<Condition, ParseError> {
        let field = self.expect_field_name()?;
        let op = self.parse_compare_op()?;
        let value = self.parse_value()?;
        Ok(Condition { field, op, value })
    }

    fn parse_compare_op(&mut self) -> Result<CompareOp, ParseError> {
        match self.peek() {
            Token::Eq => { self.advance(); Ok(CompareOp::Eq) }
            Token::Neq => { self.advance(); Ok(CompareOp::Neq) }
            Token::Gt => { self.advance(); Ok(CompareOp::Gt) }
            Token::Lt => { self.advance(); Ok(CompareOp::Lt) }
            Token::Gte => { self.advance(); Ok(CompareOp::Gte) }
            Token::Lte => { self.advance(); Ok(CompareOp::Lte) }
            Token::Keyword(Keyword::Like) => { self.advance(); Ok(CompareOp::Like) }
            Token::Keyword(Keyword::In) => { self.advance(); Ok(CompareOp::In) }
            _ => Err(ParseError(format!("expected comparison operator, got {:?}", self.peek()))),
        }
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        match self.peek() {
            Token::StringLit(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Value::String(s))
            }
            Token::NumberLit(n) => {
                let n = n;
                self.advance();
                Ok(Value::Number(n))
            }
            Token::IntegerLit(n) => {
                let n = n;
                self.advance();
                Ok(Value::Integer(n))
            }
            Token::Dash => {
                // Negative number: - followed by number
                self.advance();
                match self.peek() {
                    Token::NumberLit(n) => {
                        self.advance();
                        Ok(Value::Number(-n))
                    }
                    Token::IntegerLit(n) => {
                        self.advance();
                        Ok(Value::Integer(-n))
                    }
                    _ => Err(ParseError(format!(
                        "expected number after '-', got {:?}",
                        self.peek()
                    ))),
                }
            }
            Token::LParen => {
                self.advance();
                let mut items = Vec::new();
                if !matches!(self.peek(), Token::RParen) {
                    items.push(self.parse_value()?);
                    while self.check_comma() {
                        self.advance();
                        items.push(self.parse_value()?);
                    }
                }
                self.expect_token(&Token::RParen)?;
                Ok(Value::List(items))
            }
            _ => Err(ParseError(format!("expected value, got {:?}", self.peek()))),
        }
    }

    fn parse_order_by(&mut self) -> Result<OrderBy, ParseError> {
        let field = self.expect_field_name()?;
        let descending = if self.check_keyword(Keyword::Desc) {
            self.advance();
            true
        } else if self.check_keyword(Keyword::Asc) {
            self.advance();
            false
        } else {
            false // default ascending
        };
        Ok(OrderBy { field, descending })
    }

    fn parse_assignments(&mut self) -> Result<Vec<Assignment>, ParseError> {
        let mut assignments = vec![self.parse_assignment()?];
        while self.check_comma() {
            self.advance();
            assignments.push(self.parse_assignment()?);
        }
        Ok(assignments)
    }

    fn parse_assignment(&mut self) -> Result<Assignment, ParseError> {
        let field = self.expect_field_name()?;
        self.expect_token(&Token::Eq)?;
        let value = self.parse_value()?;
        Ok(Assignment { field, value })
    }

    // ── Token helpers ──

    fn peek(&self) -> Token {
        self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn check_keyword(&self, kw: Keyword) -> bool {
        matches!(self.peek(), Token::Keyword(k) if k == kw)
    }

    fn check_comma(&self) -> bool {
        matches!(self.peek(), Token::Comma)
    }

    fn check_pipe(&self) -> bool {
        matches!(self.peek(), Token::Pipe)
    }

    fn eat_semicolon(&mut self) {
        if matches!(self.peek(), Token::Semicolon) {
            self.advance();
        }
    }

    fn expect_keyword(&mut self, kw: Keyword) -> Result<(), ParseError> {
        if self.check_keyword(kw) {
            self.advance();
            Ok(())
        } else {
            Err(ParseError(format!("expected {:?}, got {:?}", kw, self.peek())))
        }
    }

    fn expect_string(&mut self) -> Result<String, ParseError> {
        match self.peek() {
            Token::StringLit(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            _ => Err(ParseError(format!("expected string literal, got {:?}", self.peek()))),
        }
    }

    fn expect_u32(&mut self) -> Result<u32, ParseError> {
        match self.peek() {
            Token::IntegerLit(n) if n >= 0 => {
                self.advance();
                Ok(n as u32)
            }
            _ => Err(ParseError(format!("expected positive integer, got {:?}", self.peek()))),
        }
    }

    fn expect_f32(&mut self) -> Result<f32, ParseError> {
        match self.peek() {
            Token::NumberLit(n) => {
                self.advance();
                Ok(n as f32)
            }
            Token::IntegerLit(n) => {
                self.advance();
                Ok(n as f32)
            }
            _ => Err(ParseError(format!("expected number, got {:?}", self.peek()))),
        }
    }

    fn expect_token(&mut self, expected: &Token) -> Result<(), ParseError> {
        let tok = self.peek();
        if std::mem::discriminant(&tok) == std::mem::discriminant(expected) {
            self.advance();
            Ok(())
        } else {
            Err(ParseError(format!("expected {:?}, got {:?}", expected, tok)))
        }
    }

    fn expect_ident_eq(&mut self, name: &str) -> Result<(), ParseError> {
        match self.peek() {
            Token::Ident(ref s) if s.eq_ignore_ascii_case(name) => {
                self.advance();
                Ok(())
            }
            // Also accept keywords that match field names
            Token::Keyword(kw) if format!("{:?}", kw).eq_ignore_ascii_case(name) => {
                self.advance();
                Ok(())
            }
            _ => Err(ParseError(format!("expected '{}', got {:?}", name, self.peek()))),
        }
    }

    fn expect_field_name(&mut self) -> Result<String, ParseError> {
        match self.peek() {
            Token::Ident(ref name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            Token::Keyword(kw) => {
                // Allow keywords as field names (e.g., "layer", "confidence", "relation")
                let name = format!("{:?}", kw).to_lowercase();
                self.advance();
                Ok(name)
            }
            _ => Err(ParseError(format!("expected field name, got {:?}", self.peek()))),
        }
    }
}

/// Convenience: parse a string directly into a Statement.
pub fn parse(input: &str) -> Result<Statement, Box<dyn std::error::Error>> {
    let mut lexer = crate::lexer::Lexer::new(input);
    let tokens = lexer.tokenise()?;
    let mut parser = Parser::new(tokens);
    Ok(parser.parse()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ══════════════════════════════════════════════════════════════
    // LIFECYCLE STATEMENTS
    // ══════════════════════════════════════════════════════════════

    // ── EXTRACT ──

    #[test]
    fn parse_extract_minimal() {
        let stmt = parse(
            r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";"#,
        )
        .unwrap();
        match stmt {
            Statement::Extract {
                model,
                output,
                components,
                layers,
            } => {
                assert_eq!(model, "google/gemma-3-4b-it");
                assert_eq!(output, "gemma3-4b.vindex");
                assert!(components.is_none());
                assert!(layers.is_none());
            }
            _ => panic!("expected Extract"),
        }
    }

    #[test]
    fn parse_extract_with_components_and_layers() {
        let stmt = parse(
            r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "out.vindex" COMPONENTS FFN_GATE, FFN_DOWN, FFN_UP, EMBEDDINGS LAYERS 0-33;"#,
        )
        .unwrap();
        match stmt {
            Statement::Extract {
                components, layers, ..
            } => {
                let c = components.unwrap();
                assert_eq!(c.len(), 4);
                assert_eq!(c[0], Component::FfnGate);
                assert_eq!(c[1], Component::FfnDown);
                assert_eq!(c[2], Component::FfnUp);
                assert_eq!(c[3], Component::Embeddings);
                let l = layers.unwrap();
                assert_eq!(l.start, 0);
                assert_eq!(l.end, 33);
            }
            _ => panic!("expected Extract"),
        }
    }

    #[test]
    fn parse_extract_attn_components() {
        let stmt = parse(
            r#"EXTRACT MODEL "m" INTO "o" COMPONENTS ATTN_OV, ATTN_QK;"#,
        )
        .unwrap();
        match stmt {
            Statement::Extract { components, .. } => {
                let c = components.unwrap();
                assert_eq!(c.len(), 2);
                assert_eq!(c[0], Component::AttnOv);
                assert_eq!(c[1], Component::AttnQk);
            }
            _ => panic!("expected Extract"),
        }
    }

    // ── COMPILE ──

    #[test]
    fn parse_compile_current_safetensors() {
        let stmt = parse(
            r#"COMPILE CURRENT INTO MODEL "edited/" FORMAT safetensors;"#,
        )
        .unwrap();
        match stmt {
            Statement::Compile {
                vindex,
                output,
                format,
            } => {
                assert!(matches!(vindex, VindexRef::Current));
                assert_eq!(output, "edited/");
                assert_eq!(format, Some(OutputFormat::Safetensors));
            }
            _ => panic!("expected Compile"),
        }
    }

    #[test]
    fn parse_compile_path_gguf() {
        let stmt = parse(
            r#"COMPILE "gemma3.vindex" INTO MODEL "out/" FORMAT gguf;"#,
        )
        .unwrap();
        match stmt {
            Statement::Compile {
                vindex,
                output,
                format,
            } => {
                assert!(matches!(vindex, VindexRef::Path(ref p) if p == "gemma3.vindex"));
                assert_eq!(output, "out/");
                assert_eq!(format, Some(OutputFormat::Gguf));
            }
            _ => panic!("expected Compile"),
        }
    }

    #[test]
    fn parse_compile_no_format() {
        let stmt = parse(
            r#"COMPILE CURRENT INTO MODEL "out/";"#,
        )
        .unwrap();
        match stmt {
            Statement::Compile { format, .. } => assert!(format.is_none()),
            _ => panic!("expected Compile"),
        }
    }

    // ── DIFF ──

    #[test]
    fn parse_diff_two_paths() {
        let stmt = parse(
            r#"DIFF "a.vindex" "b.vindex";"#,
        )
        .unwrap();
        match stmt {
            Statement::Diff { a, b, .. } => {
                assert!(matches!(a, VindexRef::Path(ref p) if p == "a.vindex"));
                assert!(matches!(b, VindexRef::Path(ref p) if p == "b.vindex"));
            }
            _ => panic!("expected Diff"),
        }
    }

    #[test]
    fn parse_diff_with_current() {
        let stmt = parse(r#"DIFF "gemma3-4b.vindex" CURRENT;"#).unwrap();
        match stmt {
            Statement::Diff {
                a: VindexRef::Path(p),
                b: VindexRef::Current,
                ..
            } => assert_eq!(p, "gemma3-4b.vindex"),
            _ => panic!("expected Diff"),
        }
    }

    #[test]
    fn parse_diff_with_limit() {
        let stmt = parse(
            r#"DIFF "a.vindex" "b.vindex" LIMIT 20;"#,
        )
        .unwrap();
        match stmt {
            Statement::Diff { limit, .. } => assert_eq!(limit, Some(20)),
            _ => panic!("expected Diff"),
        }
    }

    // ── USE ──

    #[test]
    fn parse_use_vindex() {
        let stmt = parse(r#"USE "gemma3-4b.vindex";"#).unwrap();
        match stmt {
            Statement::Use {
                target: UseTarget::Vindex(path),
            } => assert_eq!(path, "gemma3-4b.vindex"),
            _ => panic!("expected Use Vindex"),
        }
    }

    #[test]
    fn parse_use_model() {
        let stmt = parse(r#"USE MODEL "google/gemma-3-4b-it";"#).unwrap();
        match stmt {
            Statement::Use {
                target: UseTarget::Model { id, auto_extract },
            } => {
                assert_eq!(id, "google/gemma-3-4b-it");
                assert!(!auto_extract);
            }
            _ => panic!("expected Use Model"),
        }
    }

    #[test]
    fn parse_use_model_auto_extract() {
        let stmt = parse(r#"USE MODEL "google/gemma-3-4b-it" AUTO_EXTRACT;"#).unwrap();
        match stmt {
            Statement::Use {
                target: UseTarget::Model { auto_extract, .. },
            } => assert!(auto_extract),
            _ => panic!("expected Use Model AUTO_EXTRACT"),
        }
    }

    // ══════════════════════════════════════════════════════════════
    // QUERY STATEMENTS
    // ══════════════════════════════════════════════════════════════

    // ── WALK ──

    #[test]
    fn parse_walk_minimal() {
        let stmt = parse(r#"WALK "The capital of France is";"#).unwrap();
        match stmt {
            Statement::Walk {
                prompt,
                top,
                layers,
                mode,
                compare,
            } => {
                assert_eq!(prompt, "The capital of France is");
                assert!(top.is_none());
                assert!(layers.is_none());
                assert!(mode.is_none());
                assert!(!compare);
            }
            _ => panic!("expected Walk"),
        }
    }

    #[test]
    fn parse_walk_with_top() {
        let stmt = parse(r#"WALK "The capital of France is" TOP 5;"#).unwrap();
        match stmt {
            Statement::Walk { top, .. } => assert_eq!(top, Some(5)),
            _ => panic!("expected Walk"),
        }
    }

    #[test]
    fn parse_walk_full_options() {
        let stmt = parse(
            r#"WALK "prompt" TOP 5 LAYERS 25-33 MODE hybrid COMPARE;"#,
        )
        .unwrap();
        match stmt {
            Statement::Walk {
                top,
                layers,
                mode,
                compare,
                ..
            } => {
                assert_eq!(top, Some(5));
                let l = layers.unwrap();
                assert_eq!(l.start, 25);
                assert_eq!(l.end, 33);
                assert_eq!(mode, Some(WalkMode::Hybrid));
                assert!(compare);
            }
            _ => panic!("expected Walk"),
        }
    }

    #[test]
    fn parse_walk_mode_pure() {
        let stmt = parse(r#"WALK "x" MODE pure;"#).unwrap();
        match stmt {
            Statement::Walk { mode, .. } => assert_eq!(mode, Some(WalkMode::Pure)),
            _ => panic!("expected Walk"),
        }
    }

    #[test]
    fn parse_walk_mode_dense() {
        let stmt = parse(r#"WALK "x" MODE dense;"#).unwrap();
        match stmt {
            Statement::Walk { mode, .. } => assert_eq!(mode, Some(WalkMode::Dense)),
            _ => panic!("expected Walk"),
        }
    }

    #[test]
    fn parse_walk_layers_all() {
        let stmt = parse(r#"WALK "x" LAYERS ALL;"#).unwrap();
        match stmt {
            Statement::Walk { layers, .. } => assert!(layers.is_none()),
            _ => panic!("expected Walk"),
        }
    }

    // ── SELECT ──

    #[test]
    fn parse_select_star() {
        let stmt = parse("SELECT * FROM EDGES;").unwrap();
        match stmt {
            Statement::Select { fields, .. } => {
                assert_eq!(fields.len(), 1);
                assert!(matches!(fields[0], Field::Star));
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_named_fields() {
        let stmt = parse(
            r#"SELECT entity, relation, target, confidence FROM EDGES WHERE entity = "France" ORDER BY confidence DESC LIMIT 10;"#,
        )
        .unwrap();
        match stmt {
            Statement::Select {
                fields,
                conditions,
                order,
                limit,
                ..
            } => {
                assert_eq!(fields.len(), 4);
                assert!(matches!(&fields[0], Field::Named(n) if n == "entity"));
                assert!(matches!(&fields[1], Field::Named(n) if n == "relation"));
                assert!(matches!(&fields[2], Field::Named(n) if n == "target"));
                assert_eq!(conditions.len(), 1);
                assert_eq!(conditions[0].field, "entity");
                assert!(matches!(conditions[0].op, CompareOp::Eq));
                let ord = order.unwrap();
                assert!(ord.descending);
                assert_eq!(limit, Some(10));
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_multiple_conditions() {
        let stmt = parse(
            r#"SELECT * FROM EDGES WHERE relation = "capital-of" AND confidence > 0.5;"#,
        )
        .unwrap();
        match stmt {
            Statement::Select { conditions, .. } => {
                assert_eq!(conditions.len(), 2);
                assert_eq!(conditions[0].field, "relation");
                assert!(matches!(conditions[0].op, CompareOp::Eq));
                assert!(matches!(conditions[1].op, CompareOp::Gt));
                assert!(matches!(conditions[1].value, Value::Number(n) if (n - 0.5).abs() < 0.01));
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_by_layer_and_feature() {
        let stmt = parse(
            "SELECT * FROM EDGES WHERE layer = 26 AND feature = 9515;",
        )
        .unwrap();
        match stmt {
            Statement::Select { conditions, .. } => {
                assert_eq!(conditions.len(), 2);
                assert!(matches!(conditions[0].value, Value::Integer(26)));
                assert!(matches!(conditions[1].value, Value::Integer(9515)));
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_nearest() {
        let stmt = parse(
            r#"SELECT entity, target, distance FROM EDGES NEAREST TO "Mozart" AT LAYER 26 LIMIT 20;"#,
        )
        .unwrap();
        match stmt {
            Statement::Select {
                nearest, limit, ..
            } => {
                let n = nearest.unwrap();
                assert_eq!(n.entity, "Mozart");
                assert_eq!(n.layer, 26);
                assert_eq!(limit, Some(20));
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_no_where() {
        let stmt = parse("SELECT * FROM EDGES LIMIT 5;").unwrap();
        match stmt {
            Statement::Select {
                conditions, limit, ..
            } => {
                assert!(conditions.is_empty());
                assert_eq!(limit, Some(5));
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_order_asc() {
        let stmt = parse(
            "SELECT * FROM EDGES ORDER BY layer ASC;",
        )
        .unwrap();
        match stmt {
            Statement::Select { order, .. } => {
                let ord = order.unwrap();
                assert!(!ord.descending);
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_order_default_asc() {
        let stmt = parse(
            "SELECT * FROM EDGES ORDER BY layer;",
        )
        .unwrap();
        match stmt {
            Statement::Select { order, .. } => {
                let ord = order.unwrap();
                assert!(!ord.descending);
            }
            _ => panic!("expected Select"),
        }
    }

    // ── DESCRIBE ──

    #[test]
    fn parse_describe_minimal() {
        let stmt = parse(r#"DESCRIBE "France";"#).unwrap();
        match stmt {
            Statement::Describe {
                entity,
                layer,
                relations_only,
            } => {
                assert_eq!(entity, "France");
                assert!(layer.is_none());
                assert!(!relations_only);
            }
            _ => panic!("expected Describe"),
        }
    }

    #[test]
    fn parse_describe_at_layer() {
        let stmt = parse(r#"DESCRIBE "Mozart" AT LAYER 26;"#).unwrap();
        match stmt {
            Statement::Describe { entity, layer, .. } => {
                assert_eq!(entity, "Mozart");
                assert_eq!(layer, Some(26));
            }
            _ => panic!("expected Describe"),
        }
    }

    #[test]
    fn parse_describe_relations_only() {
        let stmt = parse(r#"DESCRIBE "France" RELATIONS ONLY;"#).unwrap();
        match stmt {
            Statement::Describe {
                relations_only, ..
            } => assert!(relations_only),
            _ => panic!("expected Describe"),
        }
    }

    #[test]
    fn parse_describe_layer_and_relations_only() {
        let stmt = parse(
            r#"DESCRIBE "France" AT LAYER 26 RELATIONS ONLY;"#,
        )
        .unwrap();
        match stmt {
            Statement::Describe {
                layer,
                relations_only,
                ..
            } => {
                assert_eq!(layer, Some(26));
                assert!(relations_only);
            }
            _ => panic!("expected Describe"),
        }
    }

    // ── EXPLAIN ──

    #[test]
    fn parse_explain_minimal() {
        let stmt = parse(r#"EXPLAIN WALK "The capital of France is";"#).unwrap();
        match stmt {
            Statement::Explain {
                prompt,
                layers,
                verbose,
            } => {
                assert_eq!(prompt, "The capital of France is");
                assert!(layers.is_none());
                assert!(!verbose);
            }
            _ => panic!("expected Explain"),
        }
    }

    #[test]
    fn parse_explain_with_layers_and_verbose() {
        let stmt = parse(
            r#"EXPLAIN WALK "prompt" LAYERS 24-33 VERBOSE;"#,
        )
        .unwrap();
        match stmt {
            Statement::Explain {
                layers, verbose, ..
            } => {
                let l = layers.unwrap();
                assert_eq!(l.start, 24);
                assert_eq!(l.end, 33);
                assert!(verbose);
            }
            _ => panic!("expected Explain"),
        }
    }

    // ══════════════════════════════════════════════════════════════
    // MUTATION STATEMENTS
    // ══════════════════════════════════════════════════════════════

    // ── INSERT ──

    #[test]
    fn parse_insert_minimal() {
        let stmt = parse(
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John Coyle", "lives-in", "Colchester");"#,
        )
        .unwrap();
        match stmt {
            Statement::Insert {
                entity,
                relation,
                target,
                layer,
                confidence,
            } => {
                assert_eq!(entity, "John Coyle");
                assert_eq!(relation, "lives-in");
                assert_eq!(target, "Colchester");
                assert!(layer.is_none());
                assert!(confidence.is_none());
            }
            _ => panic!("expected Insert"),
        }
    }

    #[test]
    fn parse_insert_with_layer_and_confidence() {
        let stmt = parse(
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John", "occupation", "engineer") AT LAYER 26 CONFIDENCE 0.8;"#,
        )
        .unwrap();
        match stmt {
            Statement::Insert {
                layer, confidence, ..
            } => {
                assert_eq!(layer, Some(26));
                let c = confidence.unwrap();
                assert!((c - 0.8).abs() < 0.01);
            }
            _ => panic!("expected Insert"),
        }
    }

    // ── DELETE ──

    #[test]
    fn parse_delete_single_condition() {
        let stmt = parse(
            r#"DELETE FROM EDGES WHERE entity = "outdated_fact";"#,
        )
        .unwrap();
        match stmt {
            Statement::Delete { conditions } => {
                assert_eq!(conditions.len(), 1);
                assert_eq!(conditions[0].field, "entity");
            }
            _ => panic!("expected Delete"),
        }
    }

    #[test]
    fn parse_delete_multiple_conditions() {
        let stmt = parse(
            r#"DELETE FROM EDGES WHERE entity = "John Coyle" AND relation = "lives-in";"#,
        )
        .unwrap();
        match stmt {
            Statement::Delete { conditions } => assert_eq!(conditions.len(), 2),
            _ => panic!("expected Delete"),
        }
    }

    #[test]
    fn parse_delete_by_layer() {
        let stmt = parse(
            r#"DELETE FROM EDGES WHERE entity = "outdated" AND layer = 26;"#,
        )
        .unwrap();
        match stmt {
            Statement::Delete { conditions } => {
                assert_eq!(conditions.len(), 2);
                assert!(matches!(conditions[1].value, Value::Integer(26)));
            }
            _ => panic!("expected Delete"),
        }
    }

    // ── UPDATE ──

    #[test]
    fn parse_update_single_set() {
        let stmt = parse(
            r#"UPDATE EDGES SET target = "London" WHERE entity = "John Coyle" AND relation = "lives-in";"#,
        )
        .unwrap();
        match stmt {
            Statement::Update { set, conditions } => {
                assert_eq!(set.len(), 1);
                assert_eq!(set[0].field, "target");
                assert!(matches!(&set[0].value, Value::String(s) if s == "London"));
                assert_eq!(conditions.len(), 2);
            }
            _ => panic!("expected Update"),
        }
    }

    // ── MERGE ──

    #[test]
    fn parse_merge_minimal() {
        let stmt = parse(r#"MERGE "source.vindex";"#).unwrap();
        match stmt {
            Statement::Merge {
                source,
                target,
                conflict,
            } => {
                assert_eq!(source, "source.vindex");
                assert!(target.is_none());
                assert!(conflict.is_none());
            }
            _ => panic!("expected Merge"),
        }
    }

    #[test]
    fn parse_merge_into_with_conflict() {
        let stmt = parse(
            r#"MERGE "medical.vindex" INTO "gemma3.vindex" ON CONFLICT HIGHEST_CONFIDENCE;"#,
        )
        .unwrap();
        match stmt {
            Statement::Merge {
                source,
                target,
                conflict,
            } => {
                assert_eq!(source, "medical.vindex");
                assert_eq!(target.as_deref(), Some("gemma3.vindex"));
                assert_eq!(conflict, Some(ConflictStrategy::HighestConfidence));
            }
            _ => panic!("expected Merge"),
        }
    }

    #[test]
    fn parse_merge_keep_source() {
        let stmt = parse(
            r#"MERGE "a.vindex" INTO "b.vindex" ON CONFLICT KEEP_SOURCE;"#,
        )
        .unwrap();
        match stmt {
            Statement::Merge { conflict, .. } => {
                assert_eq!(conflict, Some(ConflictStrategy::KeepSource));
            }
            _ => panic!("expected Merge"),
        }
    }

    #[test]
    fn parse_merge_keep_target() {
        let stmt = parse(
            r#"MERGE "a.vindex" INTO "b.vindex" ON CONFLICT KEEP_TARGET;"#,
        )
        .unwrap();
        match stmt {
            Statement::Merge { conflict, .. } => {
                assert_eq!(conflict, Some(ConflictStrategy::KeepTarget));
            }
            _ => panic!("expected Merge"),
        }
    }

    // ══════════════════════════════════════════════════════════════
    // INTROSPECTION STATEMENTS
    // ══════════════════════════════════════════════════════════════

    // ── SHOW ──

    #[test]
    fn parse_show_relations_minimal() {
        let stmt = parse("SHOW RELATIONS;").unwrap();
        match stmt {
            Statement::ShowRelations {
                layer,
                with_examples,
            } => {
                assert!(layer.is_none());
                assert!(!with_examples);
            }
            _ => panic!("expected ShowRelations"),
        }
    }

    #[test]
    fn parse_show_relations_with_examples() {
        let stmt = parse("SHOW RELATIONS WITH EXAMPLES;").unwrap();
        match stmt {
            Statement::ShowRelations { with_examples, .. } => assert!(with_examples),
            _ => panic!("expected ShowRelations"),
        }
    }

    #[test]
    fn parse_show_relations_at_layer() {
        let stmt = parse("SHOW RELATIONS AT LAYER 26;").unwrap();
        match stmt {
            Statement::ShowRelations { layer, .. } => assert_eq!(layer, Some(26)),
            _ => panic!("expected ShowRelations"),
        }
    }

    #[test]
    fn parse_show_layers_minimal() {
        let stmt = parse("SHOW LAYERS;").unwrap();
        match stmt {
            Statement::ShowLayers { range } => assert!(range.is_none()),
            _ => panic!("expected ShowLayers"),
        }
    }

    #[test]
    fn parse_show_layers_with_range() {
        let stmt = parse("SHOW LAYERS RANGE 0-10;").unwrap();
        match stmt {
            Statement::ShowLayers { range } => {
                let r = range.unwrap();
                assert_eq!(r.start, 0);
                assert_eq!(r.end, 10);
            }
            _ => panic!("expected ShowLayers"),
        }
    }

    #[test]
    fn parse_show_features_minimal() {
        let stmt = parse("SHOW FEATURES 26;").unwrap();
        match stmt {
            Statement::ShowFeatures {
                layer,
                conditions,
                limit,
            } => {
                assert_eq!(layer, 26);
                assert!(conditions.is_empty());
                assert!(limit.is_none());
            }
            _ => panic!("expected ShowFeatures"),
        }
    }

    #[test]
    fn parse_show_features_with_where_and_limit() {
        let stmt = parse(
            r#"SHOW FEATURES 26 WHERE relation = "capital-of" LIMIT 5;"#,
        )
        .unwrap();
        match stmt {
            Statement::ShowFeatures {
                layer,
                conditions,
                limit,
            } => {
                assert_eq!(layer, 26);
                assert_eq!(conditions.len(), 1);
                assert_eq!(limit, Some(5));
            }
            _ => panic!("expected ShowFeatures"),
        }
    }

    #[test]
    fn parse_show_models() {
        let stmt = parse("SHOW MODELS;").unwrap();
        assert!(matches!(stmt, Statement::ShowModels));
    }

    // ── STATS ──

    #[test]
    fn parse_stats_no_path() {
        let stmt = parse("STATS;").unwrap();
        assert!(matches!(stmt, Statement::Stats { vindex: None }));
    }

    #[test]
    fn parse_stats_with_path() {
        let stmt = parse(r#"STATS "gemma3.vindex";"#).unwrap();
        match stmt {
            Statement::Stats { vindex } => assert_eq!(vindex.as_deref(), Some("gemma3.vindex")),
            _ => panic!("expected Stats"),
        }
    }

    #[test]
    fn parse_stats_no_semicolon() {
        let stmt = parse("STATS").unwrap();
        assert!(matches!(stmt, Statement::Stats { vindex: None }));
    }

    // ══════════════════════════════════════════════════════════════
    // PIPE OPERATOR
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn parse_pipe_walk_to_explain() {
        let stmt = parse(
            r#"WALK "The capital of France is" TOP 5 |> EXPLAIN WALK "The capital of France is";"#,
        )
        .unwrap();
        match stmt {
            Statement::Pipe { left, right } => {
                assert!(matches!(*left, Statement::Walk { .. }));
                assert!(matches!(*right, Statement::Explain { .. }));
            }
            _ => panic!("expected Pipe"),
        }
    }

    // ══════════════════════════════════════════════════════════════
    // COMPARISON OPERATORS
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn parse_select_neq() {
        let stmt = parse(
            r#"SELECT * FROM EDGES WHERE relation != "morphological";"#,
        )
        .unwrap();
        match stmt {
            Statement::Select { conditions, .. } => {
                assert!(matches!(conditions[0].op, CompareOp::Neq));
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_gte_lte() {
        let stmt = parse(
            "SELECT * FROM EDGES WHERE layer >= 20 AND layer <= 30;",
        )
        .unwrap();
        match stmt {
            Statement::Select { conditions, .. } => {
                assert!(matches!(conditions[0].op, CompareOp::Gte));
                assert!(matches!(conditions[1].op, CompareOp::Lte));
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_like() {
        let stmt = parse(
            r#"SELECT * FROM EDGES WHERE entity LIKE "Fran%";"#,
        )
        .unwrap();
        match stmt {
            Statement::Select { conditions, .. } => {
                assert!(matches!(conditions[0].op, CompareOp::Like));
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_in() {
        let stmt = parse(
            r#"SELECT * FROM EDGES WHERE entity IN ("France", "Germany");"#,
        )
        .unwrap();
        match stmt {
            Statement::Select { conditions, .. } => {
                assert!(matches!(conditions[0].op, CompareOp::In));
                if let Value::List(items) = &conditions[0].value {
                    assert_eq!(items.len(), 2);
                } else {
                    panic!("expected list value");
                }
            }
            _ => panic!("expected Select"),
        }
    }

    // ══════════════════════════════════════════════════════════════
    // COMMENTS AND WHITESPACE
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn parse_with_leading_comment() {
        let stmt = parse("-- This is a comment\nSTATS;").unwrap();
        assert!(matches!(stmt, Statement::Stats { .. }));
    }

    #[test]
    fn parse_with_trailing_comment() {
        let stmt = parse("STATS; -- trailing comment").unwrap();
        assert!(matches!(stmt, Statement::Stats { .. }));
    }

    #[test]
    fn parse_multiline_statement() {
        let stmt = parse(
            "SELECT *\n  FROM EDGES\n  WHERE layer = 26\n  LIMIT 5;",
        )
        .unwrap();
        match stmt {
            Statement::Select {
                conditions, limit, ..
            } => {
                assert_eq!(conditions.len(), 1);
                assert_eq!(limit, Some(5));
            }
            _ => panic!("expected Select"),
        }
    }

    // ══════════════════════════════════════════════════════════════
    // ERROR CASES
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn parse_error_unknown_statement() {
        assert!(parse("FOOBAR;").is_err());
    }

    #[test]
    fn parse_error_walk_missing_prompt() {
        assert!(parse("WALK TOP 5;").is_err());
    }

    #[test]
    fn parse_error_select_missing_from() {
        assert!(parse(r#"SELECT * WHERE entity = "x";"#).is_err());
    }

    #[test]
    fn parse_error_insert_missing_values() {
        assert!(parse("INSERT INTO EDGES (entity, relation, target);").is_err());
    }

    #[test]
    fn parse_error_show_invalid_noun() {
        assert!(parse("SHOW FOOBAR;").is_err());
    }

    #[test]
    fn parse_error_empty_input() {
        assert!(parse("").is_err());
    }

    #[test]
    fn parse_error_comment_only() {
        assert!(parse("-- just a comment").is_err());
    }

    // ══════════════════════════════════════════════════════════════
    // FULL DEMO SCRIPT FROM SPEC — every statement parses
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn parse_demo_script_act1() {
        // EXTRACT + USE + STATS
        parse(r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";"#).unwrap();
        parse(r#"USE "gemma3-4b.vindex";"#).unwrap();
        parse("STATS;").unwrap();
    }

    #[test]
    fn parse_demo_script_act2() {
        // SHOW + DESCRIBE + SELECT
        parse("SHOW RELATIONS WITH EXAMPLES;").unwrap();
        parse(r#"DESCRIBE "France";"#).unwrap();
        parse(
            r#"SELECT entity, target, confidence FROM EDGES WHERE relation = "capital-of" ORDER BY confidence DESC LIMIT 10;"#,
        )
        .unwrap();
    }

    #[test]
    fn parse_demo_script_act3() {
        // WALK + EXPLAIN
        parse(r#"WALK "The capital of France is" TOP 5 COMPARE;"#).unwrap();
        parse(r#"EXPLAIN WALK "The capital of France is";"#).unwrap();
    }

    #[test]
    fn parse_demo_script_act4() {
        // INSERT + DESCRIBE + WALK
        parse(r#"WALK "Where does John Coyle live?" TOP 5;"#).unwrap();
        parse(
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John Coyle", "lives-in", "Colchester");"#,
        )
        .unwrap();
        parse(r#"DESCRIBE "John Coyle";"#).unwrap();
        parse(r#"WALK "Where does John Coyle live?" TOP 5;"#).unwrap();
    }

    #[test]
    fn parse_demo_script_act5() {
        // DIFF + COMPILE
        parse(r#"DIFF "gemma3-4b.vindex" CURRENT;"#).unwrap();
        parse(r#"COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;"#).unwrap();
    }
}
