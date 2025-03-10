from datasets import load_dataset


class DailyDialogCleaner:
    def __init__(self, output_file="cleaned_files/cleaned_dailydialog.txt"):
        self.dataset = load_dataset("li2017dailydialog/daily_dialog", split="train")
        self.output_file = output_file

    def clean_and_save(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            for conversation in self.dataset["dialog"]:
                for utterance in conversation:
                    f.write(utterance.strip() + " <EOS>\n")
                f.write("\n")  # Separate conversations with a blank line
        print(f"Cleaned data saved to {self.output_file}")


if __name__ == "__main__":
    cleaner = DailyDialogCleaner()
    cleaner.clean_and_save()
