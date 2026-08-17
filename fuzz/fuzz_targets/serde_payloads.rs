#![no_main]
use libfuzzer_sys::fuzz_target;
use shap_rs::{Background, Explanation, FeatureMetadata, OutputMetadata, UncertainExplanation};

fuzz_target!(|data: &[u8]| {
    if let Ok(value) = serde_json::from_slice::<Explanation>(data) { let _ = value.validate(); }
    if let Ok(value) = serde_json::from_slice::<UncertainExplanation>(data) { let _ = value.validate(); }
    if let Ok(value) = serde_json::from_slice::<Background>(data) { let _ = value.validate(); }
    if let Ok(value) = serde_json::from_slice::<FeatureMetadata>(data) { let _ = value.validate(); }
    if let Ok(value) = serde_json::from_slice::<OutputMetadata>(data) { let _ = value.validate(); }
});
