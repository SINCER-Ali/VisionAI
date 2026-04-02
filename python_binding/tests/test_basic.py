import numpy as np
import visionai

def test_import():
    assert visionai is not None


def test_vector_basic():
    v = visionai.PyVector([1.0, 2.0, 3.0])
    assert v.len() == 3
    assert v.to_list() == [1.0, 2.0, 3.0]
    assert v.sum() == 6.0
    assert v.mean() == 2.0


def test_matrix_basic():
    m = visionai.PyMatrix(2, 2, [1.0, 2.0, 3.0, 4.0])
    assert m.rows() == 2
    assert m.cols() == 2
    assert m.to_list() == [[1.0, 2.0], [3.0, 4.0]]

    t = m.transpose()
    assert t.rows() == 2
    assert t.cols() == 2
    assert t.to_list() == [[1.0, 3.0], [2.0, 4.0]]


def test_linear_regression_train_predict():
    model = visionai.create_model("linear_regression", {"input_size": 2})
    x = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    y = [[5.0], [8.0], [11.0]]

    visionai.train(model, x, y, epochs=200, lr=0.01)
    pred = visionai.predict(model, [4.0, 5.0])

    assert isinstance(pred, list)
    assert len(pred) == 1


def test_mlp_train_predict_xor():
    model = visionai.create_model("mlp", {"layer_sizes": [2, 4, 2]})

    x = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    y = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ]

    visionai.train(model, x, y, epochs=50, lr=0.01)
    pred = visionai.predict(model, [0.0, 1.0])

    assert isinstance(pred, list)
    assert len(pred) == 2


def test_save_and_load_linear_regression(tmp_path):
    model = visionai.LinearRegression(2)
    visionai.train(model, [[1.0, 2.0]], [[5.0]], epochs=10, lr=0.01)

    file_path = tmp_path / "linear_model.txt"
    visionai.save(model, str(file_path))

    loaded = visionai.load(str(file_path))
    pred = visionai.predict(loaded, [1.0, 2.0])

    assert isinstance(pred, list)
    assert len(pred) == 1


def test_numpy_conversion_vector():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    v = visionai.PyVector(arr.tolist())
    assert v.to_list() == [1.0, 2.0, 3.0]


def test_numpy_conversion_matrix():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    m = visionai.PyMatrix(2, 2, arr.flatten().tolist())
    assert m.rows() == 2
    assert m.cols() == 2
    assert m.to_list() == [[1.0, 2.0], [3.0, 4.0]]