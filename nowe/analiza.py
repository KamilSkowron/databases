# file = open("dest_ip.txt", encoding="utf8")
# plik = file.read()
# x = plik.split("},")
# for i in x:
#     i += "}"
#     with open('dest_ip_fix.txt', 'a', encoding="utf-8") as the_file:
#         the_file.write(i + "\n")

import ast
from collections import Counter

f = open("dest_ip_fix.txt", "r", encoding="utf8")
cities = []
for line in f:
    line = ast.literal_eval(line)
    if "country" in line:
        cities.append(line["country"])

    print(line['query'])


