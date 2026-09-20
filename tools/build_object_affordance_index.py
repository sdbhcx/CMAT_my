"""Build only from sources that pass the same audit used by the report tool."""
from tools.audit_multi_affordance_data import parser, run


if __name__ == '__main__':
    cli = parser()
    cli.add_argument('--index-path', required=True, help='Output JSON; canonical artifacts live beside it')
    args = cli.parse_args()
    run(args, args.index_path)
