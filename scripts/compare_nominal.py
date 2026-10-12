#!/usr/bin/env python
"""Independent nominal checkpoint evaluation and read-only artifact comparison."""
import argparse
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--training-runs', nargs='+', type=Path, required=True)
    audit.add_argument('--out', type=Path, required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--algorithm', choices=['tdmpc', 'tdmpc2', 'uncontrolled'], required=True)
    ev.add_argument('--training-run', type=Path)
    ev.add_argument('--checkpoint', type=Path)
    ev.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    ev.add_argument('--role', choices=['candidate', 'best_validation', 'final', 'reference'], default='candidate')
    ev.add_argument('--episodes', type=int, default=10)
    ev.add_argument('--base-seed', type=int, default=0)
    ev.add_argument('--device', default='cuda:0')
    ev.add_argument('--environment-steps', type=int)
    ev.add_argument('--reference', type=Path)
    ev.add_argument('--selection', type=Path)
    ev.add_argument('--out', type=Path, required=True)
    select = sub.add_parser('select')
    select.add_argument('--evaluations', nargs='+', type=Path, required=True)
    select.add_argument('--out', type=Path, required=True)
    report = sub.add_parser('report')
    report.add_argument('--evaluations', nargs='*', type=Path, default=[])
    report.add_argument('--training-runs', nargs='*', type=Path, default=[])
    report.add_argument('--out', type=Path, required=True)
    report.add_argument('--wandb-mode', choices=['disabled', 'offline', 'online'], default='disabled')
    report.add_argument('--wandb-project', default='fluidgym-compare')
    report.add_argument('--wandb-entity')
    args = ap.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    from fluidgym_rl.comparison.schema import write_json, select_best
    try:
        if args.command == 'audit':
            from fluidgym_rl.comparison.audit import audit_run
            write_json(args.out, [audit_run(p) for p in args.training_runs])
        elif args.command == 'select':
            select_best(args.evaluations, args.out)
        elif args.command == 'report':
            from fluidgym_rl.comparison.report import build_report
            build_report(args.evaluations, args.training_runs, args.out,
                         wandb_mode=args.wandb_mode, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity)
        else:
            args.out.mkdir(parents=True, exist_ok=False)
            try:
                from fluidgym_rl.comparison.evaluate import evaluate_to_directory
                evaluate_to_directory(args, args.out)
            except BaseException as exc:
                (args.out / 'traceback.txt').write_text(traceback.format_exc())
                write_json(args.out / 'failure.json', {'status': 'failed', 'error': repr(exc)})
                raise
        print(f'Completed: {args.out}')
        return 0
    except BaseException as exc:
        traceback.print_exc()
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    sys.exit(main())
