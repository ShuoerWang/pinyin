import os

folder_path = "./data"

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r',encoding="utf-8") as file:
            lines = file.readlines()

        # Remove empty lines
        lines = [line for line in lines if line.strip()]

        with open(file_path, 'w',encoding="utf-8") as file:
            file.writelines(lines)