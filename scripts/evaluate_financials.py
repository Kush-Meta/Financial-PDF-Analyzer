"""Run the dated benchmark; exit nonzero on failed/error cases, never count skips."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.evaluate import run, rescore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['offline', 'live', 'model'], default='offline')
    parser.add_argument('--only', nargs='+', choices=['financial', 'document', 'routing', 'abstention'])
    parser.add_argument('--output', default='evaluation-results/benchmark.json')
    parser.add_argument('--rescore', help='Re-score a saved report after reviewed document-source label corrections; no model/network calls.')
    args = parser.parse_args()
    if args.rescore:
        report = rescore(json.loads(Path(args.rescore).read_text()))
        output = Path(args.output)
        if output.resolve() == Path(args.rescore).resolve():
            parser.error('Use a different output path to preserve the original report.')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report['summary'], indent=2))
        return int(any(r['status'] in ('failed', 'error') for r in report['results']))
    llm = None
    index = None
    index_failure = None
    config = {}
    def retriever_factory():
        nonlocal index, index_failure
        if index_failure is not None:
            raise RuntimeError('Index construction previously failed in this evaluation run.') from index_failure
        if index is None:
            from sample_document import sample_pages
            from services import build_index, make_retriever, load_reranker
            print('Building the public Apple filing index with installed local models…', flush=True)
            try:
                vector_index = build_index(sample_pages(), 'mxbai-embed-large', base)
                encoder = load_reranker('cross-encoder/ms-marco-MiniLM-L-6-v2')
                index = make_retriever(vector_index, 8, encoder, 4)
            except Exception as exc:
                index_failure = exc
                raise
        return index
    if args.mode == 'model':
        # Evaluation must not trigger background model downloads.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        from services import make_llm
        base = os.environ.get('OLLAMA_BASE', 'http://localhost:11434')
        model = os.environ.get('FINREAD_MODEL', 'llama3')
        llm = make_llm(model, base)
        config = {'model': model, 'temperature': 0, 'embedding': 'mxbai-embed-large',
                  'reranker': 'cross-encoder/ms-marco-MiniLM-L-6-v2', 'k': 8, 'top_n': 4}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSONL checkpoints retain completed cases if the model service is interrupted.
    with path.with_suffix('.jsonl').open('w') as stream:
        def progress(row):
            stream.write(json.dumps(row) + '\n'); stream.flush()
            failures = [k for k, passed in row.get('checks', {}).items() if not passed]
            print(row['id'], row['status'], ', '.join(failures), flush=True)
        report = run(args.mode, llm, retriever_factory, progress, args.only)
    report['model_config'] = config
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))
    return int(any(r['status'] in ('failed', 'error') for r in report['results']))


if __name__ == '__main__':
    raise SystemExit(main())
