import os
import re


def process_emoticon_from_web():
    possible_char = r"[\w/_#\-:;(),.\s]"

    in_pth = os.path.join(os.getcwd(), "emoticon_web")
    out_pth = os.path.join(os.getcwd(), "emoticon_draft")
    with open(in_pth, 'r') as f_in, open(out_pth, 'w') as f_out:
        content = f_in.read()

        content = re.sub(r"\[[\w\s]+\]", "", content)

        content = re.sub(rf"</?\w+( [\w\-]+=\"{possible_char}*\")*>", "", content)
        content = re.sub(r"&gt;", ">", content)
        content = re.sub(r"&lt;", "<", content)
        content = re.sub(r"&amp;", "&", content)

        content = re.sub(r"\s+", " ", content)

        content = re.sub(r"\)\(", ") (", content)

        f_out.write(content)


if __name__ == '__main__':
    process_emoticon_from_web()
