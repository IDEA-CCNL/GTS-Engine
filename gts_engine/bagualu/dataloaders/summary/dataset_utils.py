from datasets import Dataset as hfDataset


def check_data(data: hfDataset) -> None:
    for idx in range(len(data)):
        item = data[idx]
        assert "summary" in item and isinstance(item["summary"], str)
        assert "text" in item and isinstance(item["text"], str)
