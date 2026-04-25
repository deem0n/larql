use std::path::PathBuf;

use crate::config::ExtractLevel;

#[derive(Debug, thiserror::Error)]
pub enum VindexError {
    #[error("not a directory: {0}")]
    NotADirectory(PathBuf),
    #[error("no safetensors files in {0}")]
    NoSafetensors(PathBuf),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("requires extract level '{needed}' but vindex was built at '{have}'")]
    InsufficientExtractLevel {
        needed: ExtractLevel,
        have: ExtractLevel,
    },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model error: {0}")]
    Model(#[from] larql_models::ModelError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_a_directory_includes_path() {
        let e = VindexError::NotADirectory("/tmp/missing".into());
        let s = e.to_string();
        assert!(s.contains("not a directory"), "{s}");
        assert!(s.contains("missing"), "{s}");
    }

    #[test]
    fn no_safetensors_includes_path() {
        let e = VindexError::NoSafetensors("/data/model".into());
        let s = e.to_string();
        assert!(s.contains("no safetensors"), "{s}");
        assert!(s.contains("model"), "{s}");
    }

    #[test]
    fn missing_tensor_includes_name() {
        let e = VindexError::MissingTensor("model.embed_tokens.weight".into());
        let s = e.to_string();
        assert!(s.contains("missing tensor"), "{s}");
        assert!(s.contains("model.embed_tokens.weight"), "{s}");
    }

    #[test]
    fn parse_error_includes_message() {
        let e = VindexError::Parse("unexpected token at line 5".into());
        assert!(e.to_string().contains("unexpected token at line 5"));
    }

    #[test]
    fn unsupported_dtype_includes_type() {
        let e = VindexError::UnsupportedDtype("bfloat16".into());
        let s = e.to_string();
        assert!(s.contains("unsupported dtype"), "{s}");
        assert!(s.contains("bfloat16"), "{s}");
    }

    #[test]
    fn insufficient_extract_level_shows_both_levels() {
        let e = VindexError::InsufficientExtractLevel {
            needed: ExtractLevel::Inference,
            have: ExtractLevel::Browse,
        };
        let s = e.to_string();
        assert!(s.contains("inference"), "{s}");
        assert!(s.contains("browse"), "{s}");
    }

    #[test]
    fn io_error_from_converts() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "oops");
        let e: VindexError = io.into();
        assert!(e.to_string().contains("IO error"));
    }
}
