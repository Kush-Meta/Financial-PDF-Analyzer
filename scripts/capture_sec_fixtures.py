"""Capture reviewable SEC response subsets; never regenerate expected answers."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.evaluate import load_dataset
from sec_data import SecClient, TAGS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='evaluation-results/new-fixtures')
    args=parser.parse_args()
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    dataset=load_dataset()
    client=SecClient(budget=24,deadline=150,use_cache=False)
    for ticker,reference in dataset['companies'].items():
        profile=client.resolve(ticker)
        cik=int(profile['cik'])
        facts=client.get(f'api/xbrl/companyfacts/CIK{cik:010d}.json')
        digest=lambda data: hashlib.sha256(json.dumps(data).encode()).hexdigest()
        provenance={'captured_at':datetime.now(timezone.utc).isoformat(),'kind':'subset_of_live_SEC_JSON',
                    'profile_url':f'https://data.sec.gov/submissions/CIK{cik:010d}.json',
                    'facts_url':f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json',
                    'normalized_full_profile_sha256':digest(profile),'normalized_full_facts_sha256':digest(facts),
                    'selection':'All recent 10-K/10-K/A rows; supported concept observations with the reference period end, including other accessions and units.'}
        profile={k:profile[k] for k in ('name','cik','sic','tickers','filings')}
        recent=profile['filings']['recent']
        indexes=[i for i,form in enumerate(recent['form']) if form in ('10-K','10-K/A')]
        profile['filings']={'recent':{k:[v[i] for i in indexes] for k,v in recent.items()},'files':[]}
        concepts={}
        for tag in {tag for tags in TAGS.values() for tag in tags}:
            concept=facts['facts'].get('us-gaap',{}).get(tag)
            if concept:
                concepts[tag]={**concept,'units':{unit:[f for f in rows if f.get('end')==reference['end']]
                                                  for unit,rows in concept['units'].items()}}
        result={'provenance':provenance,'profile':profile,
                'companyfacts':{'cik':facts['cik'],'entityName':facts['entityName'],'facts':{'us-gaap':concepts}}}
        target=output/f'{ticker}.json'
        # Captures are immutable within a run directory. A review decides whether
        # any new input belongs in the versioned regression set.
        with target.open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n')
        print(ticker,'captured',flush=True)


if __name__=='__main__':
    main()
