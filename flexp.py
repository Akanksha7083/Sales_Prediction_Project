def open_file():
    while True:
        file_name = input("Please enter the file name: ")
        try:

            with open(file_name, 'r') as file:
                content = file.read()
                print("File opened successfully. Here is the content:")
                print(content)
                break
        except FileNotFoundError:
            print("Error: File not found. Please enter a valid file name.")


open_file()