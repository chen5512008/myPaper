from pathlib import Path
R=Path(__file__).parent
s=(R/'run_stage2.py').read_text(encoding='utf-8')
def change(old,new):
    global s
    assert old in s,old;s=s.replace(old,new)
change('from model_v2 import RevisedAdapter','from model_region import RegionAdapter as RevisedAdapter')
change("MODES=['empirical_control_stable','empirical_summary_flex_only_stable','empirical_summary_flex_stable','empirical_summary_flex_gated_stable','chronos_summary_flex_gated_stable']","MODES=['empirical_control_region_stable','empirical_summary_flex_only_region_stable','empirical_summary_flex_gated_region_stable']")
change("'trigger':'First seven validation configurations all worse than extended LightGBM; direct summary addition worsens empirical adapter; training trajectories fluctuate. No new neural test or reserved-2026 predictions inspected.'","'trigger':'LightGBM has site identity but existing neural candidates do not; add explicit site conditioning to equalize this static information. Reserved 2026 not used for selection.'")
change("'amendment':'Second finite validation search; initial seven-config protocol and selection retained as stage1. This is a disclosed extension, not the original fixed seven-run budget.'","'amendment':'Three additional validation configurations to isolate static-site information, including a no-attention control. Prior stage protocols and selections retained; no new neural test or 2026 scoring before final selection.'")
change("'model_v2_sha256':hashlib.sha256((R/'model_v2.py').read_bytes()).hexdigest()","'model_v2_sha256':hashlib.sha256((R/'model_v2.py').read_bytes()).hexdigest(),'model_region_sha256':hashlib.sha256((R/'model_region.py').read_bytes()).hexdigest()")
change('stage2_protocol.json','region_protocol.json');change('selection_stage1.json','selection_stage2.json')
change("    return d\ndef mask_for", "    d['sid']=torch.as_tensor([SITES.index(str(s)) for s in d['site']],device=DEVICE,dtype=torch.long)\n    return d\ndef mask_for")
change("p=model(x,d['q'][a:a+BATCH])[0]", "p=model(x,d['q'][a:a+BATCH],site_ids=d['sid'][a:a+BATCH])[0]")
change("model(x,d['q'][ids],mask_for(x,scenario))[0]", "model(x,d['q'][ids],mask_for(x,scenario),d['sid'][ids])[0]")
change('START_STAGE2','START_REGION');change('SAVED_STAGE2','SAVED_REGION')
change('Best validation among disclosed stage1 and stage2; no test access for selection','Best validation among all disclosed stages including static-site controls; no test access for selection')
(R/'run_region.py').write_text(s,encoding='utf-8')
print('Wrote region runner; does not execute training or score tests.')
