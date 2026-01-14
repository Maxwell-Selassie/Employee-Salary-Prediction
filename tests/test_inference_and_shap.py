import pandas as pd
import pytest

from inference.service import InferenceService


@pytest.mark.parametrize(
    "age,years_exp",
    [
        (30, 5),
        (45, 15),
    ],
)
def test_inference_single_and_shap_shape(age: int, years_exp: int):
    service = InferenceService()
    features = {
        "Employee_age": age,
        "years_experience": years_exp,
        "Number_of_Children": 0,
        "Department": "Engineering",
        "Role": "Software Engineer",
        "performance_rating": 3,
    }
    pred = service.predict_single(features)
    assert pred > 0

    shap_values, X = service.explain_shap(pd.DataFrame([features]))
    assert shap_values.shape[1] == X.shape[1]

