#![no_main]
use libfuzzer_sys::fuzz_target;
use shap_rs::tree::adapters::{from_lightgbm_json, from_xgboost_json};

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = from_xgboost_json(text, 4, 2, vec![0.0; 2]);
        let _ = from_lightgbm_json(text, 4, 2, vec![0.0; 2]);
    }
});
