'''
    Generate datasets of pseudodivision of two multivariate polynomials with specified numbers of variables and terms\\
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


def run_maple_with_param(N: int, v: int, t: int, E: int, C: int, L: int) -> List:
    r'''Write a Maple script and run it.

    Returns a list of samples.

    Args:
        N (int): Dataset size.
        v (int): The number of variables in a polynomial.
        t (int): The number of terms in a polynomial in input polynomials.
        E (int): The maximum number of digit(s) in an exponent number in input polynomials.
        C (int): The maximum number of digit(s) in a coefficient in input polynomials.
        L (int): The maximum number of digit(s) in a coefficient in the remainder polynomial.
    '''
    maple_script = f'''
LegalCheck := proc(f::polynom, xs::list, nt::integer)
    local x:
    if nops(f) <> nt then
        return false:
    end if:
    for x in xs do
        if numboccur(f, x) = 0 then
            return false:
        end if:
    end do:
    return true:
end proc:

RandPoly := proc(xs::list, nt::integer)
    local f:
    while true do
        f := randpoly(xs, coeffs=rand(-{10 ** C - 1}..{10 ** C - 1}), expons=rand({10 ** E}), terms=nt):
        if LegalCheck(f, xs, nt) then
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

LimitCoeffDigits := proc(f::polynom, xs::list, limit::integer)
    local cs:
    if f = 0 then
        return true:
    end if:
    cs := [coeffs(f, xs)]:
    if ormap(c -> length(convert(abs(c), string)) > limit, cs) then
        return false:
    end if:
    return true:
end proc:

systems := []:
GenerateSample := proc(xs::list, nt::integer, cl::integer)
    local f, g, tmp, sys, r, m, q, ri, cot, deg_g, ltg, prem_times, legal, i, ltr, qi, qig:
    global systems:
    while true do
        f := RandPoly(xs, nt):
        g := RandPoly(xs, nt):
        if degree(f, xs[1]) < degree(g, xs[1]) then
            tmp := f:
            f := g:
            g := tmp:
        end if:
        sys := [f, g]:
        if not Duplicated(sys, systems) then
            r := expand(prem(f, g, xs[1], 'm', 'q')):
            q := expand(q):
            m := expand(m):
            if LimitCoeffDigits(m, xs, cl) and LimitCoeffDigits(q, xs, cl) and LimitCoeffDigits(r, xs, cl) then
                ri := expand(m * f):
                if not LimitCoeffDigits(ri, xs, cl) then
                    next:
                end if:
                cot := [m, ri]:
                deg_g := degree(g, xs[1]):
                ltg := lcoeff(g, xs[1]) * xs[1] ^ deg_g:
                prem_times := degree(f, xs[1]) - deg_g + 1:
                legal := true:
                for i to prem_times do
                    ltr := lcoeff(ri, xs[1]) * xs[1] ^ degree(ri, xs[1]):
                    qi := expand(normal(ltr / ltg)):
                    if not LimitCoeffDigits(qi, xs, cl) then
                        legal := false:
                        break:
                    end if:
                    cot := [op(cot), qi]:
                    qig := expand(qi * g):
                    if not LimitCoeffDigits(qig, xs, cl) then
                        legal := false:
                        break:
                    end if:
                    cot := [op(cot), qig]:
                    ri := expand(ri - qig):
                    if not LimitCoeffDigits(ri, xs, cl) then
                        legal := false:
                        break:
                    end if:
                    cot := [op(cot), ri]:
                    if degree(ri, xs[1]) < deg_g then
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
xs := [seq(cat('x', i), i=0..{v-1})]:
for i to {N} do
    GenerateSample(xs, {t}, {L});
end do:
    '''
    mpl_file_path = join(abspath(dirname(__file__)), 'v' + str(v) +
                         't' + str(t) + 'E' + str(E) + 'C' + str(C) + 'L' + str(L) + '.mpl')
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
parser.add_argument('--v', type=int, default=3,
                    help='the number of variables in a polynomial (default: 3)')
parser.add_argument('--t', type=int, default=3,
                    help='the number of terms in a polynomial in input polynomials (default: 3)')
parser.add_argument('--E', type=int, default=1,
                    help='the maximum number of digit(s) in an exponent number in input polynomials (default: 1)')
parser.add_argument('--C', type=int, default=1,
                    help='the maximum number of digit(s) in a coefficient in input polynomials (default: 1)')
parser.add_argument('--L', type=int, default=3,
                    help='the maximum number of digit(s) in a coefficient in the remainder polynomial (default: 3)')
args = parser.parse_args()

# Dataset folder
base_dir = join(abspath(dirname(__file__)), '..', '..', '..', 'dataset', 'prem', 'v' +
                str(args.v) + 't' + str(args.t) + 'E' + str(args.E) + 'C' + str(args.C) + 'L' + str(args.L))
generate_folder(base_dir)

# Generate dataset
data = run_maple_with_param(args.N, args.v, args.t, args.E, args.C, args.L)
for i in range(0, args.N * 4, 4):
    sample = PremData([data[i], data[i + 1]],
                      eval(data[i + 2]), eval(data[i + 3]))
    save(sample, join(base_dir, str(i // 4) + '.pt'))
