from test_data import montage_parameters
from EMaligner import EMaligner

def test_first_test():
    print(montage_parameters)
    mod = EMaligner.EMaligner(input_data = montage_parameters,args=[])
    mod.run()
