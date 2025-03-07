def process_and_split_conversation(file_path, train_path, test_path, train_ratio=0.9):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove "human 1:" and "human 2:"
    processed_lines = [
        line.split(":", 1)[1].strip() + "\n" for line in lines if ":" in line
    ]

    # Compute split index
    split_idx = int(len(processed_lines) * train_ratio)

    # Write train and test files
    with open(train_path, "w", encoding="utf-8") as f_train:
        f_train.writelines(processed_lines[:split_idx])

    with open(test_path, "w", encoding="utf-8") as f_test:
        f_test.writelines(processed_lines[split_idx:])


process_and_split_conversation("human_chat.txt", "train.txt", "test.txt")
