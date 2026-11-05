'''
    Generate datasets of pseudodivision of two univariate polynomials with a limit on maximum possible degree\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 18 Feb 2026
'''
from typing import List
from subprocess import check_output, CalledProcessError
from os import remove
from argparse import ArgumentParser
from os.path import join, abspath, dirname
import platform
import sys  # noqa
sys.path.append(join(abspath(dirname(__file__)), '..', '..'))  # noqa
from utils.folder import generate_folder  # noqa
from data_utils import PremData  # noqa
from torch import save


def run_maple_with_param(N: int, D: int, C: int, L: int) -> List:
    r'''Write a Maple script and run it.

    Returns a list of samples.

    Args:
        N (int): Dataset size.
        D (int): The maximum degree of a polynomial.
        C (int): The maximum number of digit(s) in a coefficient in input polynomials.
        L (int): The maximum number of digit(s) in a coefficient in the remainder polynomial.
    '''
    maple_script = f'''
LegalCheck := proc(f::polynom, x::name)
    if f = 0 or degree(f, x) = 0 then
        return false:
    end if:
    return true:
end proc:

RandPoly := proc(x::name, maxdeg::integer)
    local f:
    while true do
        f := randpoly(x, coeffs=rand(-{10 **C - 1}..{10 **C - 1}), dense, degree=rand(1..maxdeg)()):
        if LegalCheck(f, x) then
            break:
        end if:
    end do:
    return f:
end proc:

Duplicated := proc(sys::list, L::list)
    local i:
    for i from 1 to nops(L) do
        if sys = L[i] then
            return true:
        end if:
    end do:
    return false:
end proc:

LimitCoeffDigits := proc(f::polynom, x::name, limit::integer)
    local cs:
    if f = 0 then
        return true:
    end if:
    cs := [coeffs(f, x)]:
    if ormap(c -> length(convert(abs(c), string)) > limit, cs) then
        return false:
    end if:
    return true:
end proc:

systems := []:
GenerateSample := proc(maxdeg::integer, cl::integer)
    local f, g, sys, t, r, m, q, ri, cot, deg_g, ltg, prem_times, legal, i, ltr, qi, qig:
    global systems:
    while true do
        f := RandPoly(x, maxdeg):
        g := RandPoly(x, maxdeg):
        if degree(f, x) = degree(g, x) then
            next:
        end if:
        sys := sort([f, g]):
        if degree(f, x) < degree(g, x) then
            t := f:
            f := g:
            g := t:
        end if:
        if maxdeg > 2 then
            if degree(f, x) < 2 ^ (log[2](maxdeg + 1) - 1) then
                next:
            end if:
        end if:
        if not Duplicated(sys, systems) then
            r := prem(f, g, x, 'm', 'q'):
            m := expand(m):
            q := expand(q):
            if LimitCoeffDigits(m, x, cl) and LimitCoeffDigits(q, x, cl) and LimitCoeffDigits(r, x, cl) then
                ri := expand(m * f):
                if not LimitCoeffDigits(ri, x, cl) then
                    next:
                end if:
                cot := [m, ri]:
                deg_g := degree(g, x):
                ltg := lcoeff(g, x) * x ^ deg_g:
                prem_times := degree(f, x) - deg_g + 1:
                legal := true:
                for i to prem_times do
                    ltr := lcoeff(ri, x) * x ^ degree(ri, x):
                    qi := expand(normal(ltr / ltg)):
                    if not LimitCoeffDigits(qi, x, cl) then
                        legal := false:
                        break:
                    end if:
                    cot := [op(cot), qi]:
                    qig := expand(qi * g):
                    if not LimitCoeffDigits(qig, x, cl) then
                        legal := false:
                        break:
                    end if:
                    cot := [op(cot), qig]:
                    ri := expand(ri - qig):
                    if not LimitCoeffDigits(ri, x, cl) then
                        legal := false:
                        break:
                    end if:
                    cot := [op(cot), ri]:
                    if degree(ri, x) < deg_g then
                        break:
                    end if:
                end do:
                if legal then
                    systems := [op(systems), sys]:
                    break:
                end if:
            end if:
        end if:
    end do:
    print(f);
    print("#");
    print(g);
    print("#");
    print(cat("[", StringTools[Join](map(s -> cat("'", convert(s, string), "'"), [m, q, r]), ", "), "]")):
    print("#");
    print(cat("[", StringTools[Join](map(s -> cat("'", convert(s, string), "'"), cot), ", "), "]")):
    print("#");
    return:
end proc:

randomize(0):
interface(prettyprint = 0):
for i to {N} do
    GenerateSample({D}, {L});
end do:
    '''
    mpl_file_path = join(abspath(dirname(__file__)), '_v1D' +
                         str(D) + 'C' + str(C) + 'L' + str(L) + '.mpl')
    with open(mpl_file_path, 'w', encoding='utf-8') as file:
        file.write(maple_script)

    try:
        executor = 'cmaple' if platform.system() == 'Windows' else 'maple'
        command = f'{executor} -q {mpl_file_path}'
        output = check_output(command, shell=True, text=True)
        remove(mpl_file_path)

        muldata = output.strip().replace(' ', '').replace('\n', '').replace(
            '\\', '').replace('"', '').replace(' ', '').split('#')[:-1]
        return muldata

    except CalledProcessError as err:
        raise NotImplementedError(f'Error running Maple: {err}')


parser = ArgumentParser()
parser.add_argument('--N', type=int, default=100000,
                    help='dataset size (default: 100,000)')
parser.add_argument('--D', type=int, default=3,
                    help='the maximum degree of a polynomial (default: 3)')
parser.add_argument('--C', type=int, default=1,
                    help='the maximum number of digit(s) in a coefficient in input polynomials (default: 1)')
parser.add_argument('--L', type=int, default=3,
                    help='the maximum number of digit(s) in a coefficient in the remainder polynomial (default: 3)')
args = parser.parse_args()

# Dataset folder
base_dir = join(abspath(dirname(__file__)), '..', '..', '..', 'dataset',
                'prem', '_v1D' + str(args.D) + 'C' + str(args.C) + 'L' + str(args.L))
generate_folder(base_dir)

# Generate dataset
data = run_maple_with_param(args.N, args.D, args.C, args.L)
for i in range(0, args.N * 4, 4):
    sample = PremData([data[i], data[i + 1]],
                      eval(data[i + 2]), eval(data[i + 3]))
    save(sample, join(base_dir, str(i // 4) + '.pt'))
