from test_data import montage_parameters
from EMaligner import AssembleAndSolve

def test_first_test():
    print(montage_parameters)
    mod = AssembleAndSolve(input_data = montage_parameters,args=[])
    mod.run()
