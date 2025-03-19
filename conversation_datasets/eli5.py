import datasets


def extract_eli5_entries(
    output_file="conversation_datasets/cleaned_files/cleaned_eli5_entries.txt3",
    num_entries=100_000,
):
    """
    Extract the first specified number of entries from the ELI5 dataset
    and save them to a text file with <EOS> at the end of each utterance.

    Args:
        output_file (str): Path to the output text file
        num_entries (int): Number of entries to extract
    """
    print(f"Loading ELI5 dataset...")

    # Load the ELI5 dataset
    eli5_dataset = datasets.load_dataset("rexarski/eli5_category", split="train")

    # Determine the number of entries to extract (minimum of available and requested)
    num_to_extract = min(num_entries, len(eli5_dataset))
    print(
        f"Extracting {num_to_extract} entries out of {len(eli5_dataset)} available entries..."
    )

    # Extract and write entries to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_to_extract):
            # Get the question and concatenate all answers
            entry = eli5_dataset[i]
            question = entry["title"]

            # Write the question followed by <EOS> and a newline
            f.write(f"{question} <EOS>\n")

            # Process each answer
            for answer in entry["answers"]["text"]:
                f.write(f"{answer} <EOS>\n")

            # Progress indicator every 1000 entries
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} entries...")

    print(f"Extraction complete. Output saved to {output_file}")


if __name__ == "__main__":
    extract_eli5_entries()
