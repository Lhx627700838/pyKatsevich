from tests.t01 import test_pipeline as t01
from tests.t03 import test_pipeline as t03
from tests.Naeotom import test_pipeline as Naeotom
from tests.Naeotom import only_reconstruct_pipeline as Naeotom_reco
from tests.Naeotom_curve import only_differentiate as only_differentiate
from tests.Naeotom_curve import test_pipeline as Naeotom_curve
from tests.Naeotom_curve import test_config 
from tests.Naeotom_curve import only_reconstruct_pipeline_astra as Naeotom_reco_curve
if __name__=="__main__":
    
    yaml_settings_file = "Naeotom_spine_curve.yaml"
    #test_config(yaml_settings_file)
    Naeotom_curve(yaml_settings_file)
    #only_differentiate(yaml_settings_file)
    #Naeotom_reco_curve(yaml_settings_file)
    '''
    yaml_settings_file = "test03.yaml"
    t03(yaml_settings_file)
    '''
