# save something
FILE_NAME = "output26.txt"

def save_to_output_file(text, what_saved):

    with open(FILE_NAME, "a") as f:
        f.write(text)
    print(f"[DEBUG] {what_saved} written to `{FILE_NAME}` file")
