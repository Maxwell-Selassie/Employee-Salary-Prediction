from preprocessing.preprocessing_pipeline import PreprocessingPipeline


def test_preprocessing_splits_and_schema():
    pipeline = PreprocessingPipeline()
    train, dev, test = pipeline.fit_transform()

    # basic sanity on shapes and schema consistency
    assert len(train) > 0 and len(dev) > 0 and len(test) > 0
    assert list(train.columns) == list(dev.columns) == list(test.columns)

