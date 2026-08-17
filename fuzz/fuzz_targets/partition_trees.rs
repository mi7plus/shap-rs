#![no_main]
use libfuzzer_sys::fuzz_target;
use shap_rs::explainers::{FeaturePartition, PartitionTree};

fuzz_target!(|data: &[u8]| {
    if let Ok(value) = serde_json::from_slice::<FeaturePartition>(data) { let _ = value.validate(); }
    if let Ok(value) = serde_json::from_slice::<PartitionTree>(data) { let _ = value.validate(); }
});
