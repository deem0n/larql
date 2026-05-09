//! Synthetic-safetensors fixtures for the streaming-extract MoE arms.
//!
//! Hand-built, deterministic, in-process — no HuggingFace, no large
//! model downloads. Each fixture writes a tempdir tree (`config.json` +
//! `tokenizer.json` + `model.safetensors`) shaped like a real
//! architecture and drives [`larql_vindex::build_vindex_streaming`]
//! against it. The point is to exercise the per-format arms inside
//! `extract::streaming::stages::*` that the dense Llama fixture in
//! `test_vindex.rs` doesn't reach:
//!
//! - `gate_vectors::write_gate_vectors` — standard MoE arm
//! - `down_meta::write_down_meta` — standard MoE arm
//! - `router_weights::write_router_weights` — whole body (early-returns
//!   on dense; only fires when `is_moe`)
//! - `index_json::write_index_json` — MoE config branch + has-experts
//!   per-layer tracking
//!
//! Single Mixtral-shaped happy path is enough to flip all four files
//! into "MoE arm exercised" territory.

use std::collections::HashMap;
use std::path::Path;

use larql_vindex::{
    build_vindex_streaming, ExtractLevel, Q4kWriteOptions, QuantFormat, SilentBuildCallbacks,
    StorageDtype, WriteWeightsOptions,
};

/// Build a tiny Mixtral-shaped model (block-sparse MoE FFN with
/// `num_experts` experts per layer). Deterministic per-tensor ramps
/// so two runs against the same dims produce identical vindexes.
///
/// Returns the in-memory tokenizer so callers can drive
/// `build_vindex_streaming` without re-reading the JSON file.
fn write_synthetic_mixtral_model(
    model_dir: &Path,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    vocab: usize,
) -> larql_vindex::tokenizers::Tokenizer {
    std::fs::create_dir_all(model_dir).unwrap();

    let config = serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
        "num_local_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |name: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };

    // Embedding + final norm.
    push("model.embed_tokens.weight", vec![vocab, hidden]);
    push("model.norm.weight", vec![hidden]);

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        // Standard Llama-style attention.
        push(&format!("{lp}.self_attn.q_proj.weight"), vec![hidden, hidden]);
        push(&format!("{lp}.self_attn.k_proj.weight"), vec![hidden, hidden]);
        push(&format!("{lp}.self_attn.v_proj.weight"), vec![hidden, hidden]);
        push(&format!("{lp}.self_attn.o_proj.weight"), vec![hidden, hidden]);
        push(&format!("{lp}.input_layernorm.weight"), vec![hidden]);
        push(
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        // Block-sparse MoE: router + per-expert gate (w1) / down (w2) / up (w3).
        push(
            &format!("{lp}.block_sparse_moe.gate.weight"),
            vec![num_experts, hidden],
        );
        for e in 0..num_experts {
            let ep = format!("{lp}.block_sparse_moe.experts.{e}");
            push(&format!("{ep}.w1.weight"), vec![intermediate, hidden]);
            push(&format!("{ep}.w2.weight"), vec![hidden, intermediate]);
            push(&format!("{ep}.w3.weight"), vec![intermediate, hidden]);
        }
    }

    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, &None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), serialized).unwrap();

    // Minimal BPE tokenizer — enough for safetensors-backed extracts
    // that don't need to encode strings.
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

#[test]
fn streaming_extract_mixtral_exercises_moe_arms() {
    // Tiny dims chosen so each FFN row pads to a clean Q4_K boundary if
    // the test ever extends to quant=Q4K. For now we extract f32 at
    // Browse level — covers gate / down_meta / router / index_json.
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_mixtral_model(
        &model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-synthetic",
        &output_dir,
        5, // down_top_k
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        Q4kWriteOptions::default(),
        false, // drop_gate_vectors
        &mut cb,
    )
    .expect("streaming extract on mixtral fixture");

    // ── Outputs the MoE arms must produce ───────────────────────
    assert!(output_dir.join("gate_vectors.bin").exists());
    assert!(output_dir.join("router_weights.bin").exists(),
            "MoE arm must write router_weights.bin (router_weights.rs whole body)");
    assert!(output_dir.join("embeddings.bin").exists());
    assert!(output_dir.join("down_meta.bin").exists());
    assert!(output_dir.join("index.json").exists());

    // ── index.json carries MoE config (index_json.rs MoE branch) ──
    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    let model_cfg = config.model_config.expect("model_config present");
    let moe = model_cfg.moe.expect("MoE config recorded");
    assert_eq!(moe.num_experts, num_experts);
    assert_eq!(moe.top_k, num_experts_per_tok);

    // ── layer_infos record per-expert geometry (gate_vectors arm) ──
    assert_eq!(config.layers.len(), num_layers);
    for layer_info in &config.layers {
        assert_eq!(layer_info.num_experts, Some(num_experts));
        assert_eq!(layer_info.num_features_per_expert, Some(intermediate));
        // Total = num_experts × intermediate.
        assert_eq!(layer_info.num_features, num_experts * intermediate);
    }

    // ── router_weights.bin shape: per-layer router (+ optional bias) ──
    // Each router is `num_experts × hidden` f32 = 4 floats × 4 bytes = 16 B.
    // Two layers → ≥ 32 B (more if biases happened to be present).
    let router_bytes = std::fs::metadata(output_dir.join("router_weights.bin"))
        .unwrap()
        .len();
    let min_expected = (num_layers * num_experts * hidden * 4) as u64;
    assert!(
        router_bytes >= min_expected,
        "router_weights.bin {router_bytes} B < expected {min_expected} B"
    );
}
