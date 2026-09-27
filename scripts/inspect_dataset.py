from lyra.data.dataset import (
    WORKFLOWS,
    load_typed_decisions,
)


def main():
    print("=" * 80)
    print("LYRA — PHASE 0 DATASET INSPECTION")
    print("=" * 80)

    print("\nAvailable workflows:")
    for workflow in WORKFLOWS:
        print(f"  - {workflow}")

    print("\nLoading customer_service/train...")

    dataset = load_typed_decisions(
        workflow="customer_service",
        split="train",
    )

    print("\nDataset:")
    print(dataset)

    print("\nNumber of examples:")
    print(len(dataset))

    print("\nFeatures:")
    print(dataset.features)

    print("\nColumn names:")
    print(dataset.column_names)

    print("\nFirst example:")
    print("-" * 80)

    example = dataset[0]

    for key, value in example.items():
        print(f"\n[{key}]")
        print(value)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
