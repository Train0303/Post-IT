import re

string = './gender_classifier/dataset/woman\\001967.jpg'

p = re.compile("[0-9]+")
m = p.findall(string)
print(m)