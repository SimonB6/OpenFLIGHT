import re

file_name = "output_wan_0"
f = open(file_name+'.log')
out_f = open(file_name+'.csv', 'w')
text = f.read()
pattern = r'----------------------------------EXP.+'
rtext = re.split(pattern, text)[1:]                     # [protocol]
for i in range(len(rtext)):
    rtext[i] = rtext[i].split("+-----+")                # [protocol, record]
for i in range(len(rtext)):
    for j in range(len(rtext[i])):
        rtext[i][j] = rtext[i][j].strip().split("\n")           # [protocol, record, line]
        if rtext[i][j] == [""]: continue
        result = ""
        for k in range(len(rtext[i][j])):
            tag, item = rtext[i][j][k].split(":")
            if tag == "desc":
                item = item.strip()
            else:
                item = re.sub(r'[^\.0-9]+', '', item).strip()
            result += item + ","
        out_f.write(result[:-1] + "\n")
