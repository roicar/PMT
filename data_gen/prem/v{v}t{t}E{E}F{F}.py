'''
    Generate datasets of pseudodivision of two multivariate polynomials with specified numbers of variables & terms and coefficients in a finite field\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 13 Feb 2026
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


def run_maple_with_param(N: int, v: int, t: int, E: int, F: int) -> List:
    r'''Write a Maple script and run it.

    Returns a list of samples.

    Args:
        N (int): Dataset size.
        v (int): The number of variables in a polynomial.
        t (int): The number of terms in a polynomial in input polynomials.
        E (int): The maximum number of digit(s) in an exponent number in input polynomials.
        F (int): The modulus in a finite field.
    '''
    maple_script = f'''
LegalCheck := proc(p::polynom, xs::list, nt::integer)
    local x:
    if nops(p) <> nt then
        return false:
    end if:
    for x in xs do
        if numboccur(p, x) = 0 then
            return false:
        end if:
    end do:
    # Main variable x0
    if degree(p, xs[1]) = 0 then
        return false:
    end if:
    return true:
end proc:

RandPoly := proc(xs::list, nt::integer)
    local p:
    while true do
        p := randpoly(xs, coeffs=rand(), expons=rand({10 ** E}), terms=nt):
        if LegalCheck(p, xs, nt) then
            break:
        end if:
    end do:
    return p:
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

LimitCoeffDigits := proc(p::polynom, xs::list, limit::integer)
    local cs:
    if p = 0 then
        return false:
    end if:
    cs := [coeffs(p, xs)]:
    if ormap(c -> length(convert(abs(c), string)) > limit, cs) then
        return false:
    end if:
    return true:
end proc:

systems := []:
GenerateSample := proc(xs::list, nt::integer, cl::integer)
    local p1, p2, sys, pt, r, m, q:
    global systems:
    while true do
        p1 := RandPoly(xs, nt):
        p2 := RandPoly(xs, nt):
        sys := sort([p1, p2]):
        if degree(p1, xs[1]) < degree(p2, xs[1]) then
            pt := p1:
            p1 := p2:
            p2 := pt:
        end if:
        if not Duplicated(sys, systems) then
            r := expand(prem(p1, p2, x0, 'm', 'q')):
            if LimitCoeffDigits(r, xs, cl) then
                systems := [op(systems), sys]:
                break:
            end if:
        end if:
    end do:
    print(p1);
    print("#");
    print(p2);
    print("#");
    print(r);
    print("#");
    print(expand(m));
    print("#");
    print(expand(q));
    print("#");
    return:
end proc:

randomize(0):
interface(prettyprint = 0):
xs := [seq(cat('x', i), i=0..{v - 1})]:
for i to {N} do
    GenerateSample(xs, {t}, {F});
end do:
    '''
    mpl_file_path = join(abspath(dirname(__file__)), 'v' + str(v) + 't' + str(t) + 'E' + str(E) + 'F' + str(F) + '.mpl')
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
parser.add_argument('--v', type=int, default=2,
                    help='the number of variables in a polynomial (default: 2)')
parser.add_argument('--t', type=int, default=2,
                    help='the number of terms in a polynomial in input polynomials (default: 2)')
parser.add_argument('--E', type=int, default=1,
                    help='the maximum number of digit(s) in an exponent number in input polynomials (default: 1)')
parser.add_argument('--F', type=int, default=7,
                    help='the modulus in a finite field (default: 7)')
args = parser.parse_args()

# Dataset folder
base_dir = join(abspath(dirname(__file__)), '..', '..', '..', 'dataset', 'prem', 'v' + str(args.v) + 't' + str(args.t) + 'E' + str(args.E) + 'F' + str(args.F))
generate_folder(base_dir)

# Generate dataset
data = run_maple_with_param(args.N, args.v, args.t, args.E, args.F)
for i in range(0, args.N * 3, 3):
    sample = PremData([data[i], data[i + 1]], data[i + 2])
    save(sample, join(base_dir, str(i // 3) + '.pt'))
