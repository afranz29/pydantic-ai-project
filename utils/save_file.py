# save something

def save_to_output_file(text):

    with open("output19.txt", "a") as f:
        f.write(text)
    print("[DEBUG] Report prompt written to report_prompt.txt")
