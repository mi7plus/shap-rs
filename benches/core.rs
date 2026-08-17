use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{array, Array2, ArrayView2, Axis};
use shap_rs::{
    explainers::{ExactExplainer, KernelExplainer, TreeExplainer},
    interactions::{ExactInteractionExplainer, TreeInteractionExplainer},
    plot::svg::{global_bar, SvgOptions},
    Background, Explainer, Explanation, FixedMasker, FnModel, MissingBranch, Node,
    ParallelExplainerExt, Tree, TreeEnsemble,
};

fn sum_model(x: ArrayView2<'_, f64>) -> shap_rs::Result<Array2<f64>> {
    Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1)))
}

fn tree_model() -> TreeEnsemble {
    let tree = Tree::new(
        vec![
            Node::Split {
                feature: 0,
                threshold: 0.0,
                left: 1,
                right: 2,
                missing: MissingBranch::Left,
                cover: 10.0,
            },
            Node::Leaf {
                values: vec![1.0],
                cover: 4.0,
            },
            Node::Leaf {
                values: vec![5.0],
                cover: 6.0,
            },
        ],
        0,
        1,
    )
    .unwrap();
    TreeEnsemble::new(vec![(tree, 1.0)], vec![0.0]).unwrap()
}

fn tree_forest(count: usize) -> TreeEnsemble {
    let tree = tree_model().trees()[0].0.clone();
    TreeEnsemble::new((0..count).map(|_| (tree.clone(), 1.0)).collect(), vec![0.0]).unwrap()
}

fn benchmarks(c: &mut Criterion) {
    let sample = array![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
    c.bench_function("coalition_evaluation_exact_6", |b| {
        b.iter(|| {
            ExactExplainer::new(
                FnModel::new(sum_model),
                Background::new(Array2::zeros((8, 6))).unwrap(),
            )
            .explain(black_box(sample.view()))
            .unwrap()
        })
    });
    c.bench_function("kernel_wls_6", |b| {
        b.iter(|| {
            KernelExplainer::new(
                FnModel::new(sum_model),
                Background::new(Array2::zeros((8, 6))).unwrap(),
            )
            .with_nsamples(64)
            .explain(black_box(sample.view()))
            .unwrap()
        })
    });
    let tree = tree_model();
    let tree_sample = array![[2.0]];
    c.bench_function("tree_shap", |b| {
        b.iter(|| {
            TreeExplainer::new(&tree)
                .explain(black_box(tree_sample.view()))
                .unwrap()
        })
    });
    let forest = tree_forest(128);
    let tree_batch = Array2::from_shape_fn((1024, 1), |(sample, _)| sample as f64 - 512.0);
    c.bench_function("tree_shap_batch_1024x128", |b| {
        b.iter(|| {
            TreeExplainer::new(&forest)
                .explain(black_box(tree_batch.view()))
                .unwrap()
        })
    });
    c.bench_function("tree_shap_parallel_1024x128", |b| {
        b.iter(|| {
            TreeExplainer::new(&forest)
                .explain_parallel(black_box(tree_batch.view()), 64)
                .unwrap()
        })
    });
    c.bench_function("tree_interactions", |b| {
        b.iter(|| {
            TreeInteractionExplainer::new(&tree)
                .explain(black_box(tree_sample.view()))
                .unwrap()
        })
    });
    c.bench_function("exact_model_agnostic_interactions", |b| {
        b.iter(|| {
            ExactInteractionExplainer::new(
                FnModel::new(sum_model),
                FixedMasker::new(ndarray::Array1::zeros(6)).unwrap(),
            )
            .explain(black_box(sample.view()))
            .unwrap()
        })
    });
    let explanation = Explanation::new(
        ndarray::Array3::ones((32, 16, 1)),
        Array2::zeros((32, 1)),
        Array2::ones((32, 16)),
    )
    .unwrap();
    c.bench_function("explanation_json", |b| {
        b.iter(|| explanation.to_json().unwrap())
    });
    c.bench_function("global_bar_svg", |b| {
        b.iter(|| global_bar(black_box(&explanation), &SvgOptions::default()).unwrap())
    });
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
