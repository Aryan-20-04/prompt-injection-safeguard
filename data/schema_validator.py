class DatasetValidator:
    """
    Validates dataset integrity before benchmarking.
    """

    def validate(self, dataset):

        if not dataset.samples:
            raise ValueError("Dataset contains no samples")

        ids = set()

        for sample in dataset.samples:

            if sample.sample_id in ids:
                raise ValueError(f"Duplicate sample_id detected: {sample.sample_id}")

            ids.add(sample.sample_id)

            if not sample.text:
                raise ValueError(f"Sample {sample.sample_id} missing text")

            if not sample.ground_truth_labels:
                raise ValueError(
                    f"Sample {sample.sample_id} missing ground truth labels"
                )