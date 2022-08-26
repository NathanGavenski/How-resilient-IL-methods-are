from venv import create
from utils.Domains import Domain
from utils.CreateExpert import get_args, create_expert

if __name__ == '__main__':
    args = get_args()
    Domain(args.domain)
    create_expert(args, env=args.domain)