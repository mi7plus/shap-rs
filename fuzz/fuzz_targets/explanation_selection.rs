#![no_main]
use libfuzzer_sys::fuzz_target;
use shap_rs::Explanation;

fuzz_target!(|data: &[u8]| {
    if let Ok(value) = serde_json::from_slice::<Explanation>(data) {
        let indices = data.iter().take(8).map(|&v| usize::from(v)).collect::<Vec<_>>();
        let _ = value.select_samples(&indices);
        let _ = value.select_features(&indices);
        for &output in &indices { let _ = value.select_output(output); }
    }
});
