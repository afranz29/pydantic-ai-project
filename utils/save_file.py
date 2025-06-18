# save something
FILE_NAME = "output21.txt"

def save_to_output_file(text):

    with open(FILE_NAME, "a") as f:
        f.write(text)
    print(f"[DEBUG] Report prompt written to `{FILE_NAME}` file")
