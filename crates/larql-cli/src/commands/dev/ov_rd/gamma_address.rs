use std::collections::HashMap;

use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};

use super::address::{
    predict_code_from_hyperplanes, train_binary_hyperplane, AddressSupervisedGroupModel,
};
use super::basis::{WoRoundtripBasis, ZPcaBasis};
use super::metrics::argmax_usize;
use super::pq::PqCodebook;
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::stats::StaticHeadMeans;
use super::types::{HeadId, PqConfig, PromptRecord};

#[derive(Debug, Clone)]
pub(super) struct GammaProjectedAddressModel {
    pub(super) name: String,
    pub(super) source: GammaProjectionSource,
    pub(super) supervised: AddressSupervisedGroupModel,
}

impl GammaProjectedAddressModel {
    pub(super) fn selected_group_keys(&self) -> Vec<String> {
        self.supervised
            .selected_group_keys()
            .into_iter()
            .map(|key| format!("{}:{key}", self.name))
            .collect()
    }

    pub(super) fn project_layer_input(
        &self,
        layer_input: &Array2<f32>,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        match &self.source {
            GammaProjectionSource::Raw => Ok(layer_input.clone()),
            GammaProjectionSource::DiagonalAffine(map) => {
                let mut rows = Vec::with_capacity(layer_input.len());
                for row in layer_input.rows() {
                    rows.extend(
                        map.project(
                            row.as_slice().ok_or(
                                "layer input row was not contiguous during gamma projection",
                            )?,
                        ),
                    );
                }
                Ok(Array2::from_shape_vec(layer_input.raw_dim(), rows)?)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum GammaProjectionSource {
    Raw,
    DiagonalAffine(DiagonalAffineMap),
}

#[derive(Debug, Clone)]
pub(super) struct DiagonalAffineMap {
    mean_x: Vec<f32>,
    mean_y: Vec<f32>,
    slope: Vec<f32>,
}

impl DiagonalAffineMap {
    fn project(&self, row: &[f32]) -> Vec<f32> {
        row.iter()
            .enumerate()
            .map(|(dim, &x)| self.mean_y[dim] + self.slope[dim] * (x - self.mean_x[dim]))
            .collect()
    }
}

#[derive(Debug, Clone)]
struct GammaCodeSample {
    head: HeadId,
    config: PqConfig,
    position: usize,
    raw_input: Vec<f32>,
    targets: HashMap<usize, Vec<f32>>,
    codes: Vec<usize>,
}

pub(super) fn fit_gamma_projected_address_models(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    selected_groups: &[usize],
    projection_layers: &[usize],
    epochs: usize,
    lr: f32,
    l2: f32,
) -> Result<HashMap<(HeadId, PqConfig), Vec<GammaProjectedAddressModel>>, Box<dyn std::error::Error>>
{
    let samples = collect_gamma_code_samples(
        weights,
        index,
        tokenizer,
        prompts,
        heads,
        bases,
        means,
        pca_bases,
        codebooks,
        projection_layers,
        "gamma-address-fit",
    )?;
    let dim = weights.hidden_size;

    let mut samples_by_head_config: HashMap<(HeadId, PqConfig), Vec<&GammaCodeSample>> =
        HashMap::new();
    let mut samples_by_head: HashMap<HeadId, Vec<&GammaCodeSample>> = HashMap::new();
    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();
    for sample in &samples {
        samples_by_head_config
            .entry((sample.head, sample.config))
            .or_default()
            .push(sample);
        samples_by_head.entry(sample.head).or_default().push(sample);
        for (group, &code) in sample.codes.iter().enumerate() {
            let levels = 1usize << sample.config.bits_per_group;
            majority_counts
                .entry((sample.head, sample.config, group))
                .or_insert_with(|| vec![0; levels])[code] += 1;
        }
    }

    let mut maps_by_head_layer: HashMap<(HeadId, usize), DiagonalAffineMap> = HashMap::new();
    for head in heads {
        let head_samples = samples_by_head.get(head).cloned().unwrap_or_default();
        for &projection_layer in projection_layers {
            let pairs = head_samples
                .iter()
                .filter_map(|sample| {
                    sample
                        .targets
                        .get(&projection_layer)
                        .map(|target| (sample.raw_input.as_slice(), target.as_slice()))
                })
                .collect::<Vec<_>>();
            if !pairs.is_empty() {
                maps_by_head_layer.insert(
                    (*head, projection_layer),
                    fit_diagonal_affine_map(&pairs, dim),
                );
            }
        }
    }

    let mut out = HashMap::new();
    for ((head, config), _) in codebooks {
        let train_samples = samples_by_head_config
            .get(&(*head, *config))
            .cloned()
            .unwrap_or_default();
        let mut group_majority = Vec::with_capacity(config.groups);
        for group in 0..config.groups {
            group_majority.push(
                majority_counts
                    .get(&(*head, *config, group))
                    .map(|counts| argmax_usize(counts))
                    .unwrap_or(0),
            );
        }

        let mut models = Vec::new();
        let raw_rows = train_samples
            .iter()
            .map(|sample| sample.raw_input.clone())
            .collect::<Vec<_>>();
        models.push(fit_one_projected_model(
            "gamma_raw",
            GammaProjectionSource::Raw,
            &raw_rows,
            &train_samples,
            *config,
            selected_groups,
            &group_majority,
            epochs,
            lr,
            l2,
        ));

        for &projection_layer in projection_layers {
            let Some(map) = maps_by_head_layer.get(&(*head, projection_layer)).cloned() else {
                continue;
            };
            let projected_rows = train_samples
                .iter()
                .map(|sample| map.project(&sample.raw_input))
                .collect::<Vec<_>>();
            models.push(fit_one_projected_model(
                &format!("gamma_diag_post_l{projection_layer}"),
                GammaProjectionSource::DiagonalAffine(map),
                &projected_rows,
                &train_samples,
                *config,
                selected_groups,
                &group_majority,
                epochs,
                lr,
                l2,
            ));
        }

        out.insert((*head, *config), models);
    }

    Ok(out)
}

fn fit_one_projected_model(
    name: &str,
    source: GammaProjectionSource,
    rows: &[Vec<f32>],
    samples: &[&GammaCodeSample],
    config: PqConfig,
    selected_groups: &[usize],
    group_majority: &[usize],
    epochs: usize,
    lr: f32,
    l2: f32,
) -> GammaProjectedAddressModel {
    let dim = rows.first().map(Vec::len).unwrap_or(0);
    let row_refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut group_hyperplanes = vec![Vec::new(); config.groups];
    let mut group_train_accuracy = vec![0.0; config.groups];
    for &group in selected_groups {
        let mut bit_planes = Vec::with_capacity(config.bits_per_group);
        for bit in 0..config.bits_per_group {
            let labels = samples
                .iter()
                .map(|sample| ((sample.codes[group] >> bit) & 1) != 0)
                .collect::<Vec<_>>();
            bit_planes.push(train_binary_hyperplane(
                &row_refs, &labels, dim, epochs, lr, l2,
            ));
        }

        let mut correct = 0usize;
        for (row, sample) in rows.iter().zip(samples.iter()) {
            let predicted = predict_code_from_hyperplanes(row, &bit_planes);
            if predicted == sample.codes[group] {
                correct += 1;
            }
        }
        group_train_accuracy[group] = if rows.is_empty() {
            0.0
        } else {
            correct as f64 / rows.len() as f64
        };
        group_hyperplanes[group] = bit_planes;
    }

    GammaProjectedAddressModel {
        name: name.to_string(),
        source,
        supervised: AddressSupervisedGroupModel {
            groups: selected_groups.to_vec(),
            bits_per_group: config.bits_per_group,
            epochs,
            lr,
            l2,
            group_majority: group_majority.to_vec(),
            group_hyperplanes,
            group_train_accuracy,
        },
    }
}

fn fit_diagonal_affine_map(pairs: &[(&[f32], &[f32])], dim: usize) -> DiagonalAffineMap {
    let n = pairs.len().max(1) as f64;
    let mut sum_x = vec![0.0_f64; dim];
    let mut sum_y = vec![0.0_f64; dim];
    let mut sum_xx = vec![0.0_f64; dim];
    let mut sum_xy = vec![0.0_f64; dim];
    for &(x, y) in pairs {
        for dim_idx in 0..dim {
            let xi = x[dim_idx] as f64;
            let yi = y[dim_idx] as f64;
            sum_x[dim_idx] += xi;
            sum_y[dim_idx] += yi;
            sum_xx[dim_idx] += xi * xi;
            sum_xy[dim_idx] += xi * yi;
        }
    }

    let mut mean_x = vec![0.0_f32; dim];
    let mut mean_y = vec![0.0_f32; dim];
    let mut slope = vec![0.0_f32; dim];
    for dim_idx in 0..dim {
        let mx = sum_x[dim_idx] / n;
        let my = sum_y[dim_idx] / n;
        let var_x = (sum_xx[dim_idx] / n) - mx * mx;
        let cov_xy = (sum_xy[dim_idx] / n) - mx * my;
        mean_x[dim_idx] = mx as f32;
        mean_y[dim_idx] = my as f32;
        slope[dim_idx] = if var_x.abs() > 1e-12 {
            (cov_xy / var_x) as f32
        } else {
            0.0
        };
    }

    DiagonalAffineMap {
        mean_x,
        mean_y,
        slope,
    }
}

fn collect_gamma_code_samples(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    projection_layers: &[usize],
    label_prefix: &str,
) -> Result<Vec<GammaCodeSample>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }
    let max_head_layer = heads.iter().map(|head| head.layer).max().unwrap_or(0);
    let max_projection_layer = projection_layers
        .iter()
        .copied()
        .max()
        .unwrap_or(max_head_layer);
    let max_layer = max_head_layer.max(max_projection_layer);
    let projection_set = projection_layers.iter().copied().collect::<Vec<_>>();
    let mut all_samples = Vec::new();

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!(
            "  {} [{}/{}] {}",
            label_prefix,
            prompt_idx + 1,
            prompts.len(),
            label
        );
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);
        let mut prompt_samples = Vec::new();
        let mut target_rows_by_layer: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let layer_input = h.clone();
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let basis = bases.get(head).ok_or_else(|| {
                        format!("missing basis for L{}H{}", head.layer, head.head)
                    })?;
                    let head_means = means.get(head).ok_or_else(|| {
                        format!("missing means for L{}H{}", head.layer, head.head)
                    })?;
                    let pca_basis = pca_bases.get(head).ok_or_else(|| {
                        format!("missing PCA basis for L{}H{}", head.layer, head.head)
                    })?;
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    let head_codebooks = codebooks
                        .iter()
                        .filter(|((codebook_head, _), _)| codebook_head == head)
                        .collect::<Vec<_>>();
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row
                            .as_slice()
                            .ok_or("pre-W_O head row was not contiguous during gamma fit")?;
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual = values
                            .iter()
                            .zip(base.iter())
                            .map(|(&yi, &bi)| yi - bi)
                            .collect::<Vec<_>>();
                        let z = basis.residual_to_z(&residual);
                        let raw_input = layer_input
                            .row(pos)
                            .as_slice()
                            .ok_or("layer input row was not contiguous during gamma fit")?
                            .to_vec();
                        for ((_, config), codebook) in &head_codebooks {
                            let coords = pca_basis.coordinates_with_rank(&z, config.k);
                            let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                            prompt_samples.push(GammaCodeSample {
                                head: *head,
                                config: *config,
                                position: pos,
                                raw_input: raw_input.clone(),
                                targets: HashMap::new(),
                                codes,
                            });
                        }
                    }
                }
            }

            {
                let ffn = WeightFfn { weights };
                if let Some((h_new, _, _)) =
                    run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), None)
                {
                    h = h_new;
                } else {
                    remove_layer_tensors(weights, inserted);
                    return Err(format!("layer {layer} returned no output").into());
                }
            }
            remove_layer_tensors(weights, inserted);

            if projection_set.contains(&layer) {
                target_rows_by_layer.insert(
                    layer,
                    h.rows()
                        .into_iter()
                        .map(|row| row.as_slice().unwrap_or(&[]).to_vec())
                        .collect(),
                );
            }
            if layer >= max_layer {
                break;
            }
        }

        for sample in &mut prompt_samples {
            for &projection_layer in projection_layers {
                if projection_layer < sample.head.layer {
                    continue;
                }
                if let Some(rows) = target_rows_by_layer.get(&projection_layer) {
                    if let Some(target) = rows.get(sample.position) {
                        sample.targets.insert(projection_layer, target.clone());
                    }
                }
            }
        }
        all_samples.extend(prompt_samples);
    }

    Ok(all_samples)
}
