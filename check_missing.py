import sys

import melba

import pandas
import numpy



    

fpath = sys.argv[1]
table = pandas.read_csv(fpath, index_col=[0,1,2,3])

util=melba.Utils.from_file('conf.json')

combinations=util.multivariate_combinations()
print("Set size is ", len(combinations))

actual = set(table.index)

assert actual.issubset(combinations)
missing = combinations - actual

print("Missing", len(missing), "cases")
print(sorted(missing))

def to_index(fset, fcount, fsel, rep):
    conf = util.conf['multivariate_survival']
    lengths = [conf['cv_repeats'], len(conf['feature_sets']), len(conf['feature_counts']), len(conf['feature_selection'])]

    loop_sizes = numpy.cumprod(lengths[-1:0:-1])[::-1].tolist()+[1]

    fsel = conf['feature_selection'].index(fsel)
    fcount = conf['feature_counts'].index(fcount)
    fset = conf['feature_sets'].index(fset)

    indices = numpy.array([rep, fset, fcount, fsel], dtype=int)

    vals = numpy.multiply(numpy.array([rep, fset, fcount, fsel]), loop_sizes)
    index = vals.sum()
    return index


indices = [to_index(*arg) for arg in missing]
indices = sorted(indices)

for ind in indices:
    print(ind)
