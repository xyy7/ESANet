import os
import re

res = []
with open("resultsComp/STF_united_add.txt") as file:
    contents = file.readlines()
    for content in contents:
        if content.find("mIoU") != -1 and content.find("#") == -1:  # 如果是py文件,可以注释掉
            pattern = r"\d*\.\d*$"
            mat = re.findall(pattern, content)
            if mat is not None:
                mat = mat[0] if isinstance(mat, list) else mat
                res.append(float(mat))

print(f'"miou" : {str(sorted(res))},')
print(f'"miou" : {str(res)},')
