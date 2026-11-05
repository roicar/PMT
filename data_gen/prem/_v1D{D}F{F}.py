'''
    Generate datasets of pseudodivision of two univariate polynomials with a limit on maximum possible degree and coefficients in a finite field\\
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


def run_maple_with_param(N: int, D: int, F: int) -> List:
    r'''Write a Maple script and run it.

    Returns a list of samples.

    Args:
        N (int): Dataset size.
        D (int): The maximum degree of a polynomial.
        F (int): The modulus in a finite field.
    '''
    maple_script = f'''
LegalCheck := proc(f::polynom, x::name)
    if f = 0 or degree(f, x) = 0 then
        return false:
    end if:
    return true:
end proc:

RandPoly := proc(x::name, maxdeg::integer, prime::integer)
    local f:
    while true do
        f := randpoly(x, coeffs=rand(prime), dense, degree=rand(1..maxdeg)()):
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

systems := []:
GenerateSample := proc(maxdeg::integer, prime::integer)
    local f, g, sys, t, deg_g, ltg, prem_times, m, ri, cot, i, ltr, qi, qig:
    global systems:
    while true do
        f := RandPoly(x, maxdeg, prime):
        g := RandPoly(x, maxdeg, prime):
        if degree(f, x) = degree(g, x) then
            next:
        end if:
        if degree(f, x) < degree(g, x) then
            t := f:
            f := g:
            g := t:
        end if:
        sys := [f, g]:
        if not Duplicated(sys, systems) then
            deg_g := degree(g, x):
            ltg := lcoeff(g, x) * x ^ deg_g:
            prem_times := degree(f, x) - deg_g + 1:
            m := expand(lcoeff(g, x) ^ prem_times) mod prime:
            ri := expand(m * f) mod prime:
            cot := [m, ri]:
            for i to prem_times do
                ltr := lcoeff(ri, x) * x ^ degree(ri, x):
                qi := expand(normal(ltr / ltg)) mod prime:
                cot := [op(cot), qi]:
                qig := expand(qi * g) mod prime:
                cot := [op(cot), qig]:
                ri := expand(ri - qig) mod prime:
                cot := [op(cot), ri]:
                if degree(ri, x) < deg_g then
                    break:
                end if:
            end do:
            systems := [op(systems), sys]:
            break:
        end if:
    end do:
    print(f);
    print("#");
    print(g);
    print("#");
    print(cat("[", StringTools[Join](map(s -> cat("'", convert(s, string), "'"), cot), ", "), "]")):
    print("#");
    return:
end proc:

randomize(0):
interface(prettyprint = 0):
for i to {N} do
    GenerateSample({D}, {F});
end do:
    '''
    mpl_file_path = join(abspath(dirname(__file__)),
                         '_v1D' + str(D) + 'F' + str(F) + '.mpl')
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
parser.add_argument('--D', type=int, default=7,
                    help='the maximum degree of a polynomial (default: 3)')
parser.add_argument('--F', type=int, default=7,
                    help='the modulus in a finite field (default: 7)')
args = parser.parse_args()

# Dataset folder
base_dir = join(abspath(dirname(__file__)), '..', '..', '..',
                'dataset', 'prem', '_v1D' + str(args.D) + 'F' + str(args.F))
generate_folder(base_dir)

# Generate dataset
data = run_maple_with_param(args.N, args.D, args.F)
for i in range(0, args.N * 3, 3):
    sample = PremData([data[i], data[i + 1]], None, eval(data[i + 2]))
    save(sample, join(base_dir, str(i // 3) + '.pt'))
