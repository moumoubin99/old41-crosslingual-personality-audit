#!/usr/bin/env python3
"""CPU analysis replay of saved scores; never calls a model or network service."""
import argparse,hashlib,importlib,json,math,sys,time,traceback,shutil,re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/xlpt'))

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def pole_summary(pair,target):
    import summarize_pole_rater as spr
    high,low,_,label=spr.POLE_PAIRS[pair]
    shards,frozen,unit=spr.collect(ROOT/'process/results/xlpt','seed-*.jsonl',phases=(high,low))
    result=spr.summarize(shards,frozen,unit,high_phase=high,low_phase=low,pair_label=label)
    # Original execution hashes remain provenance, while publication copies have new byte hashes.
    provenance=json.loads((ROOT/'PROVENANCE.json').read_text())
    entries={x['path']:x for x in provenance['source_files']}
    checks={}
    for field,key in [('prompt_config_sha256','prompt_config'),('freeze_receipt_sha256','freeze_receipt')]:
        rel=spr.POLE_PHASES[high][key]
        entry=entries[rel]
        if digest(ROOT/rel)!=entry['public_sha256'] or entry['original_canonical_self_sha256']!=frozen[field]:
            raise ValueError('Frozen original provenance or publication bytes differ: '+rel)
        checks[rel]='original_canonical_digest_matches_rows_and_sanitized_public_copy_matches_manifest'
    for field,name in spr.SCORER_DIGEST_FILES.items():
        if field not in frozen:continue
        rel='experiments/xlpt/'+name;e=entries[rel]
        if digest(ROOT/rel)!=e['public_sha256'] or e['original_sha256']!=frozen[field]:
            raise ValueError('Published scorer provenance differs: '+name)
        checks[rel]='original_execution_digest_matches_rows_and_public_copy_matches_manifest'
    result['freeze_files_checked']=checks
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

def public_summary(target):
    import reduce_h_public as rhp
    result={'schema_version':1,'descriptive_only':True,
            'bootstrap':{'draws':rhp.N_BOOT,'seed':rhp.BOOT_SEED},'corpora':{}}
    for corpus in ('friendspersona','cped'):
        records=[json.loads(x) for x in (ROOT/'process/results/xlpt/h-public'/f'scores-{corpus}.jsonl').read_text().splitlines()]
        # Statistical reduction depends on numeric ratings and published label/cluster mappings.
        # Historical GPU/third-party-transcript provenance is documented, not re-executed here.
        result['corpora'][corpus]=rhp.reduce_corpus(corpus,records)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

def gemma_pole_scripts(target):
    # Same definitions as the archived paper/checks/gemma_script_recount.py.
    han=re.compile(r'[一-鿿㐀-䶿]');lat=re.compile(r'[A-Za-zÀ-ɏ]')
    tone=re.compile(r'[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜĀÁǍÀĒÉĚÈŌÓǑÒ]');out={}
    for pole in ('high','low'):
        n=t=z=ok=0
        files=sorted((ROOT/f'process/results/xlpt/zh-b4-extraversion-{pole}/gemma-3-12b-it').glob('seed-*.jsonl'))
        for f in files:
            for line in f.read_text().splitlines():
                for a in json.loads(line)['agent_turns']:
                    a=a if isinstance(a,str) else a.get('content','')
                    h,l=len(han.findall(a)),len(lat.findall(a));n+=1;t+=bool(tone.search(a));z+=h==0
                    ok+=(h+l>0 and h/(h+l)>=0.5)
        out[pole]={'seed_files':len(files),'turns':n,'turns_with_pinyin_tone_marks':t,'turns_with_zero_han':z,'turns_passing_rule':ok,'pass_rate':round(ok/n,4)}
    target.write_text(json.dumps(out,indent=1)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',default='recomputed',type=Path)
    ap.add_argument('--only',nargs='*',help='Optional named jobs from --list')
    ap.add_argument('--list',action='store_true')
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    jobs={
        'b0_qwen':('summarize_b0_rater','--output-json','B0_RATER_SUMMARY.json',['--shard-dir','process/results/xlpt/b0-rater/qwen3-8b']),
        'b0_aya':('summarize_b0_rater','--output-json','B0_RATER_SUMMARY_aya.json',['--shard-dir','process/results/xlpt/b0-rater/aya-expanse-8b']),
        'b0_llama':('summarize_b0_rater','--output-json','B0_RATER_SUMMARY_llama.json',['--shard-dir','process/results/xlpt/b0-rater/Llama-3.1-8B-Instruct']),
        'b0_gemma':('summarize_b0_rater','--output-json','B0_RATER_SUMMARY_gemma.json',['--shard-dir','process/results/xlpt/b0-rater/gemma-3-12b-it']),
        'b3_indicators':('analyze_b3_indicators','--output-json','B3_INDICATOR_CONTRAST.json',[]),
        'gate_readiness':('b0_gate_readiness','--output-json','B0_GATE_READINESS.json',['--gate-json','process/refine-logs/QUALIFICATION_GATE.json']+sum([['--summary-json','process/results/xlpt/'+name] for name in ['B0_RATER_SUMMARY.json','B0_RATER_SUMMARY_aya.json','B0_RATER_SUMMARY_llama.json','B0_RATER_SUMMARY_gemma.json']],[])),
        'english_inventory':('inventory_english_surface','--output-json','ENGLISH_SURFACE_INVENTORY.json',[]),
        'zh_competence':('screen_zh_competence','--output-json','ZH_G1_COMPETENCE_SCREEN.json',['--phase-root','process/results/xlpt/zh-b0-competence-repaired','--expect-shards','92']),
        'gemma_pole_scripts':('gemma','','gemma_script_recount.json',[]),
        'pole_en':('pole','en','B4_POLE_RATER_SUMMARY.json',[]),
        'pole_en_zh':('pole','zh-den','ZH_DEN_POLE_RATER_SUMMARY.json',[]),
        'pole_zh_en':('pole','bc-den','BC_DEN_POLE_RATER_SUMMARY.json',[]),
        'pole_zh_zh':('pole','bc-dzh','BC_DZH_POLE_RATER_SUMMARY.json',[]),
        'contrast_2x2':('reduce_2x2_contrast','--out','XLPT_2X2_CONTRAST.json',[]),
        'robustness_2x2':('reduce_2x2_robustness','--out','XLPT_2X2_ROBUSTNESS.json',[]),
        'audit_2x2':('reduce_2x2_audit','--out','XLPT_2X2_AUDIT.json',[]),
        'length_sensitivity':('reduce_length_sensitivity','--out','XLPT_LENGTH_SENSITIVITY.json',[]),
        'cell_surface':('describe_cell_surface','--out','XLPT_CELL_SURFACE.json',[]),
        'zh_generation':('describe_zh_b4_generation','--out','ZH_B4_POLE_GENERATION_CONTRAST.json',[]),
        'public_concordance':('public','','H_PUBLIC_SUMMARY.json',[]),
        'supplementary':('reduce_supp','--out','SUPP_SUMMARY.json',[]),
        'supplementary_common':('reduce_supp_common','--out','SUPP_COMMON_SUMMARY.json',[]),
        'supplementary_renderings':('reduce_supp_ext','--out','SUPP_EXT_SUMMARY.json',[]),
    }
    if args.list:print('\n'.join(jobs));return 0
    manifest=json.loads((ROOT/'PROVENANCE.json').read_text())
    bad=[x['path'] for x in manifest['source_files'] if digest(ROOT/x['path'])!=x['public_sha256']]
    bad += [x['path'] for x in manifest['derived_numeric_metadata'] if digest(ROOT/x['path'])!=x['public_sha256']]
    if bad:raise ValueError('Publication input hashes differ: '+str(bad))
    print('Publication input hashes verified.',flush=True)
    if args.verify_only:return 0
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for name in args.only or jobs:
        if name not in jobs:raise ValueError('Unknown job: '+name)
        module,flag,filename,extra=jobs[name];target=output/filename;started=time.time()
        try:
            if module=='pole':pole_summary(flag,target)
            elif module=='public':public_summary(target)
            elif module=='gemma':gemma_pole_scripts(target)
            elif module=='describe_zh_b4_generation':
                importlib.import_module(module).main(['--results-root',str(ROOT/'process/results/xlpt'),'--out',str(target)])
            else:
                temporary=ROOT/'recomputed'/'.working'/name/filename
                temporary.parent.mkdir(parents=True,exist_ok=True)
                module_args=['--project',str(ROOT),flag,str(temporary)]+extra
                if module in {'inventory_english_surface','screen_zh_competence'}:
                    old_argv=sys.argv;sys.argv=[module]+module_args
                    try:importlib.import_module(module).main()
                    finally:sys.argv=old_argv
                else:importlib.import_module(module).main(module_args)
                shutil.move(str(temporary),str(target))
            reports.append({'job':name,'output':filename,'ok':True,'seconds':round(time.time()-started,2)})
            print(name+': complete',flush=True)
        except Exception as exc:
            reports.append({'job':name,'ok':False,'error':str(exc),'seconds':round(time.time()-started,2)})
            traceback.print_exc()
    (output/'execution_report.json').write_text(json.dumps(reports,indent=2)+'\n')
    return int(any(not x['ok'] for x in reports))

if __name__=='__main__':raise SystemExit(main())
