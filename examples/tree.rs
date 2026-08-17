use ndarray::array;
use shap_rs::{
    explainers::TreeExplainer, Explainer, MissingBranch, Node, Result, Tree, TreeEnsemble,
};

fn main() -> Result<()> {
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
    )?;
    let model = TreeEnsemble::new(vec![(tree, 1.0)], vec![0.0])?;
    let explanation = TreeExplainer::new(&model).explain(array![[2.0]].view())?;
    println!("{:?}", explanation.values());
    Ok(())
}
