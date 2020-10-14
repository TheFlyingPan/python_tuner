import math

def convert(f1, f2):
    return round((1200 * math.log(float(f1) / float(f2), 2)), 0)