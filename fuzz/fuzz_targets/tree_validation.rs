#![no_main]
use libfuzzer_sys::fuzz_target;
use shap_rs::{Tree, TreeArrays, TreeEnsemble};

fuzz_target!(|data: &[u8]| {
    if let Ok(value) = serde_json::from_slice::<Tree>(data) { let _ = value.validate(); }
    if let Ok(value) = serde_json::from_slice::<TreeArrays>(data) { let _ = Tree::from_arrays(value); }
    if let Ok(value) = serde_json::from_slice::<TreeEnsemble>(data) { let _ = value.validate(); }
});
