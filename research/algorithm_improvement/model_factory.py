from model import ImprovedAdapter
from model_v2 import RevisedAdapter
from model_region import RegionAdapter
def build_model(scale,mean,variant,empirical):
    cls=RegionAdapter if '_region_' in variant else (RevisedAdapter if variant.endswith('_stable') else ImprovedAdapter)
    return cls(scale,mean,variant,empirical)
