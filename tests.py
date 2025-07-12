import os
contents = os.listdir("output_vedio")
for content in contents:
    full_path = os.path.join("output_vedio",content)
    os.remove(full_path)

os.rmdir("output_vedio")

