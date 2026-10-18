"""Additional seeds for the empirical reference after the planned comparison."""
from pathlib import Path
import json
from datetime import datetime,timezone
R=Path(__file__).parent
s=(R/'run.py').read_text()
a=s.index('    choices=');b=s.index('\n',a)
s=s[:a]+"    choices=[('empirical_reference',29),('empirical_reference',43)]"+s[b:]
(R/'run_empirical_seeds.py').write_text(s,encoding='utf-8')
(R/'extension_protocol.json').write_text(json.dumps({
    'recorded_utc':datetime.now(timezone.utc).isoformat(),
    'reason':'Empirical reference had lower validation loss than the foundation reference; repeat without changing architecture or optimization.',
    'status':'Exploratory extension after pooled test results were already inspected.',
    'seeds':[29,43],
    'primary_empirical_seed':17,
    'hyperparameter_changes':[],
    'spatial_extension':False
},indent=2),encoding='utf-8')
