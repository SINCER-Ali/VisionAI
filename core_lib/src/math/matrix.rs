use rand::Rng;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {

    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }


    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Matrix { rows, cols, data }
    }


    pub fn from_flat(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "Dimension mismatch");
        Matrix { rows, cols, data }
    }


    pub fn transpose(&self) -> Self {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        result
    }


    pub fn multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Matrix dimensions strictly incompatible for multiplication");

        let mut result = Matrix::new(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * result.cols + j] = sum;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*; 

    #[test]
    fn test_multiplication_simple() {
        // Matrice A (2x2)
        // [1.0, 2.0]
        // [3.0, 4.0]
        let a = Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        // Matrice B (2x2) - Identité
        // [1.0, 0.0]
        // [0.0, 1.0]
        let b = Matrix::from_flat(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

        let result = a.multiply(&b);

        // A * I doit être égal à A
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_multiplication_complexe() {
        // A (2x3)
        // [1, 2, 3]
        // [4, 5, 6]
        let a = Matrix::from_flat(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // B (3x2)
        // [7, 8]
        // [9, 1]
        // [2, 3]
        let b = Matrix::from_flat(3, 2, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]);

        // Résultat attendu (2x2)
        // [1*7 + 2*9 + 3*2,  1*8 + 2*1 + 3*3] = [31, 19]
        // [4*7 + 5*9 + 6*2,  4*8 + 5*1 + 6*3] = [85, 55]
        let result = a.multiply(&b);

        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_eq!(result.data, vec![31.0, 19.0, 85.0, 55.0]);
    }

    #[test]
    fn test_transpose() {

        let a = Matrix::from_flat(1, 3, vec![1.0, 2.0, 3.0]);

        let t = a.transpose();

        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 1);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic]
    fn test_incompatible_dimensions() {
        let a = Matrix::new(2, 2);
        let b = Matrix::new(3, 3);
        let _ = a.multiply(&b);
    }
}