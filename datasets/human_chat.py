class humanChat:
    def __init__(self):
        with open("raw_files/human_chat.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Remove "human 1:" and "human 2:"
        self.processed_lines = [
            line.split(":", 1)[1].strip() + "<EOS> \n" for line in lines if ":" in line
        ]

    def get_split_index(self, train_ratio=0.9):
        return int(len(self.processed_lines) * train_ratio)

    def fetch_data(self):
        return self.processed_lines

    def write_to_file(self, file_name):
        with open(file_name, "w", encoding="utf-8") as file:
            file.writelines(self.processed_lines)


dataset = humanChat()
dataset.write_to_file("cleaned_files/cleaned_human_chat.txt")
