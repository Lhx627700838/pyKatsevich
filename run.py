from tests.t01 import test_pipeline as t01
from tests.t03 import test_pipeline as t03
from tests.Naeotom import test_pipeline as Naeotom
from tests.Naeotom import only_reconstruct_pipeline as Naeotom_reco
if __name__=="__main__":
    yaml_settings_file = "Naeotom_spine_10919.yaml"
    Naeotom_reco(yaml_settings_file)