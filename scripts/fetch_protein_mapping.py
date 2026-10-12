"""Download a complete, counted UniProt mapping snapshot."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))
OUT = BASE/'raw/uniprot/mapping_snapshot'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2, status_forcelist=[429,500,502,503,504])))
    url = 'https://rest.uniprot.org/uniprotkb/search'
    params = {'query':'proteome:UP000006548', 'format':'tsv', 'size':500,
              'fields':'accession,gene_names,gene_oln,xref_tair,xref_interpro,xref_pfam'}
    rows, seen, pages = [], set(), []
    while url:
        response = session.get(url, params=params, timeout=(20,120))
        response.raise_for_status()
        expected = int(response.headers['x-total-results'])
        page = list(csv.DictReader(io.StringIO(response.text), delimiter='\t'))
        assert page and all(r['Entry'] not in seen for r in page)
        seen.update(r['Entry'] for r in page)
        rows.extend(page)
        (OUT/f'page_{len(pages):03d}.tsv').write_text(response.text)
        pages.append({'url':response.url,'release':response.headers.get('x-uniprot-release'), 'date':response.headers.get('date')})
        print(f'{len(rows)}/{expected}', flush=True)
        url = response.links.get('next',{}).get('url')
        params = None
    assert len(rows) == len(seen) == expected
    releases = {p['release'] for p in pages}
    assert len(releases) == 1, 'Release changed mid-download'
    target = OUT/'proteome_mapping.tsv'
    with target.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    (OUT/'verified.json').write_text(json.dumps({'rows':expected,'pages':pages,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))

if __name__ == '__main__':
    main()
