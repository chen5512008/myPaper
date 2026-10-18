from pathlib import Path
import json,datetime
R=Path(__file__).parent;V=R/'reviewer_revision';V.mkdir(exist_ok=True)
model=(R/'model.py').read_text(encoding='utf-8')
model=model.replace("if self.variant=='empirical_reference':","if self.variant.startswith('empirical'):")
model=model.replace("if self.variant=='no_availability':","if self.variant in ['no_availability','empirical_no_availability']:")
model=model.replace("if self.variant=='no_distribution_query':","if self.variant in ['no_distribution_query','empirical_no_distribution_query']:")
model=model.replace("if self.variant=='affine':","if self.variant in ['affine','empirical_affine']:")
(V/'model.py').write_text(model,encoding='utf-8')
run=(R/'run.py').read_text(encoding='utf-8')
run=run.replace('DATA=ROOT.parent','DATA=ROOT.parent.parent')
run=run.replace("choices=([('full',17),('affine',17),('no_availability',17),('no_distribution_query',17),('empirical_reference',17),('legacy_matched',17),('full',29),('full',43)] if group=='pooled' else [('full',17)])","choices=[('empirical_affine',17),('empirical_no_distribution_query',17),('empirical_no_availability',17)]")
(V/'run.py').write_text(run,encoding='utf-8')
protocol={
 'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'trigger':'User supplied review of the complete six-site manuscript',
 'status':'Exploratory follow-up on a previously inspected benchmark; not an untouched final test',
 'fixed_controls':['Unadapted empirical residual reference','Empirical shared scale seed 17','Empirical no distribution query seed 17','Empirical no availability seed 17'],
 'training':'Existing 2000-2019 training and 2020-2021 validation; original hyperparameters and seed 17 retained',
 'high_wave_diagnostics':['median MAE and bias','mean pinball','0.9 quantile loss','10-90 interval coverage and width','upper miss rate','80% interval score'],
 'calibration':'Frozen fitted predictions; 2022 used only for horizon-specific nonnegative CQR-style interval expansion; evaluation 2023-2025; no change to median or other quantiles',
 'calibration_guarantee':'No finite-sample exchangeability guarantee is claimed for the dependent time series; no conditional high-wave coverage guarantee',
 'data_need':'External wave observations for verification and a geographically and chronologically separate confirmation dataset'
}
(V/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared review follow-up protocol',V)
