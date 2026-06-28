// % de bonnes predictions
pub fn accuracy(predictions: &[usize], targets: &[usize]) -> f64 {
    assert_eq!(predictions.len(), targets.len(), "Dimensions mismatch");
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| p == t)
        .count();
    correct as f64 / predictions.len() as f64
}

// matrice de confusion : ligne = vrai label, colonne = label predit
pub fn confusion_matrix(
    predictions: &[usize],
    targets: &[usize],
    num_classes: usize,
) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0usize; num_classes]; num_classes];
    for (&pred, &target) in predictions.iter().zip(targets.iter()) {
        matrix[target][pred] += 1;
    }
    matrix
}

// precision pour une classe : tp / (tp + fp)
pub fn precision(matrix: &[Vec<usize>], class: usize) -> f64 {
    let tp = matrix[class][class];
    let fp: usize = (0..matrix.len())
        .filter(|&i| i != class)
        .map(|i| matrix[i][class])
        .sum();
    if tp + fp == 0 {
        return 0.0;
    }
    tp as f64 / (tp + fp) as f64
}

// recall pour une classe : tp / (tp + fn)
pub fn recall(matrix: &[Vec<usize>], class: usize) -> f64 {
    let tp = matrix[class][class];
    let fn_: usize = (0..matrix.len())
        .filter(|&j| j != class)
        .map(|j| matrix[class][j])
        .sum();
    if tp + fn_ == 0 {
        return 0.0;
    }
    tp as f64 / (tp + fn_) as f64
}

// f1 = moyenne harmonique precision/recall
pub fn f1_score(matrix: &[Vec<usize>], class: usize) -> f64 {
    let p = precision(matrix, class);
    let r = recall(matrix, class);
    if p + r == 0.0 {
        return 0.0;
    }
    2.0 * p * r / (p + r)
}

// f1 macro = moyenne du f1 sur toutes les classes
pub fn f1_macro(matrix: &[Vec<usize>]) -> f64 {
    let n = matrix.len();
    (0..n).map(|c| f1_score(matrix, c)).sum::<f64>() / n as f64
}

// MSE = (1/n) * sum((pred - target)^2)
pub fn mse(predictions: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(predictions.len(), targets.len(), "Dimensions mismatch");
    let n = predictions.len();
    assert!(n > 0, "Sequences vides");
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n as f64
}

// MAE = (1/n) * sum(|pred - target|)
pub fn mae(predictions: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(predictions.len(), targets.len(), "Dimensions mismatch");
    let n = predictions.len();
    assert!(n > 0, "Sequences vides");
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n as f64
}

// R^2 = 1 - SS_res / SS_tot
// 1 = parfait, 0 = equivalent a la moyenne, <0 = pire que la moyenne
pub fn r_squared(predictions: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(predictions.len(), targets.len(), "Dimensions mismatch");
    let n = targets.len();
    assert!(n > 0, "Sequences vides");
    let mean_t: f64 = targets.iter().sum::<f64>() / n as f64;
    let ss_res: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    let ss_tot: f64 = targets.iter().map(|t| (t - mean_t).powi(2)).sum();
    if ss_tot < 1e-14 {
        return 1.0;
    }
    1.0 - ss_res / ss_tot
}

// decoupe le dataset en k parties pour la cross validation
pub fn kfold_indices(n_samples: usize, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let fold_size = n_samples / k;
    (0..k)
        .map(|i| {
            let test: Vec<usize> = (i * fold_size..(i + 1) * fold_size).collect();
            let train: Vec<usize> = (0..n_samples).filter(|x| !test.contains(x)).collect();
            (train, test)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_perfect() {
        let preds = vec![0, 1, 2];
        let targets = vec![0, 1, 2];
        assert_eq!(accuracy(&preds, &targets), 1.0);
    }

    #[test]
    fn test_accuracy_zero() {
        let preds = vec![0, 0, 0];
        let targets = vec![1, 1, 1];
        assert_eq!(accuracy(&preds, &targets), 0.0);
    }

    #[test]
    fn test_confusion_matrix() {
        let preds = vec![0, 1, 2, 0];
        let targets = vec![0, 1, 1, 2];
        let m = confusion_matrix(&preds, &targets, 3);
        assert_eq!(m[0][0], 1);
        assert_eq!(m[1][1], 1);
    }

    #[test]
    fn test_f1_perfect() {
        let preds = vec![0, 1, 2];
        let targets = vec![0, 1, 2];
        let m = confusion_matrix(&preds, &targets, 3);
        assert_eq!(f1_score(&m, 0), 1.0);
    }

    #[test]
    fn test_kfold_indices() {
        let folds = kfold_indices(10, 5);
        assert_eq!(folds.len(), 5);
        assert_eq!(folds[0].1.len(), 2);
        assert_eq!(folds[0].0.len(), 8);
    }
}
