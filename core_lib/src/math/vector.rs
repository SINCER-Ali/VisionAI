use rand::Rng;

#[derive(Clone, Debug)]
pub struct Vector {
    pub len: usize,
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(len: usize) -> Self {
        Vector {
            len,
            data: vec![0.0; len],
        }
    }

    pub fn random(len: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Vector { len, data }
    }

    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Vector { len, data }
    }

    pub fn get(&self, i: usize) -> f64 {
        self.data[i]
    }

    pub fn set(&mut self, i: usize, value: f64) {
        self.data[i] = value;
    }

    pub fn add(&self, other: &Vector) -> Vector {
        assert_eq!(self.len, other.len, "Vector dimensions mismatch (add)");
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Vector::from_vec(data)
    }

    pub fn sub(&self, other: &Vector) -> Vector {
        assert_eq!(self.len, other.len, "Vector dimensions mismatch (sub)");
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Vector::from_vec(data)
    }

    pub fn scale(&self, alpha: f64) -> Vector {
        let data: Vec<f64> = self.data.iter().map(|x| alpha * x).collect();
        Vector::from_vec(data)
    }

    pub fn hadamard(&self, other: &Vector) -> Vector {
        assert_eq!(self.len, other.len, "Vector dimensions mismatch (hadamard)");
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Vector::from_vec(data)
    }

    pub fn dot(&self, other: &Vector) -> f64 {
        assert_eq!(self.len, other.len, "Vector dimensions mismatch (dot)");
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        assert!(self.len > 0, "Cannot compute mean of empty vector");
        self.sum() / self.len as f64
    }

    pub fn l2_norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn argmax(&self) -> usize {
        assert!(self.len > 0, "Cannot argmax an empty vector");
        let mut best_i = 0usize;
        let mut best_v = self.data[0];
        for i in 1..self.len {
            if self.data[i] > best_v {
                best_v = self.data[i];
                best_i = i;
            }
        }
        best_i
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub() {
        let a = Vector::from_vec(vec![10.0, 7.0, 3.0]);
        let b = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let c = a.sub(&b);
        assert_eq!(c.data, vec![9.0, 5.0, 0.0]);
    }

    #[test]
    fn test_scale() {
        let a = Vector::from_vec(vec![1.0, -2.0, 3.0]);
        let b = a.scale(2.0);
        assert_eq!(b.data, vec![2.0, -4.0, 6.0]);
    }

    #[test]
    fn test_hadamard() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let c = a.hadamard(&b);
        assert_eq!(c.data, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_dot() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn test_l2_norm() {
        let a = Vector::from_vec(vec![3.0, 4.0]);
        assert_eq!(a.l2_norm(), 5.0);
    }

    #[test]
    fn test_argmax() {
        let a = Vector::from_vec(vec![0.1, 2.0, 1.5]);
        assert_eq!(a.argmax(), 1);
    }

    #[test]
    #[should_panic]
    fn test_dot_incompatible_dimensions() {
        let a = Vector::new(3);
        let b = Vector::new(4);
        let _ = a.dot(&b);
    }
}
