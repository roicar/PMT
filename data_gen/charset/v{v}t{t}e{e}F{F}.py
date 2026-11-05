'''
    Generate datasets of computing characteristic set of two multivariate polynomials with coefficients in a finite field\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 19 Feb 2026
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
from data_utils import CharSetData  # noqa
from torch import save


def run_maple_with_param(N: int, v: int, t: int, e: int, F: int) -> List:
    r'''Write a Maple script and run it.

    Returns a list of samples.

    Args:
        N (int): Dataset size.
        v (int): The number of variables in a polynomial.
        t (int): The number of terms in a polynomial in input polynomials.
        e (int): The maximum exponent number in input polynomials.
        F (int): The modulus in a finite field.
    '''
    maple_script = f'''
read "wsolve7.mpl":

max_exec_time := 1:

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

RandPoly := proc(xs::list, nt::integer, maxexp::integer, prime::integer)
    local f:
    while true do
        f := randpoly(xs, coeffs=rand(prime), expons=rand(maxexp + 1), terms=nt):
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

systems := []:
GenerateSample := proc(xs::list, nt::integer, maxexp::integer, prime::integer)
    local f, g, sys, s, s_str, i:
    global systems:
    while true do
        f := RandPoly(xs, nt, maxexp, prime):
        g := RandPoly(xs, nt, maxexp, prime):
        if f = g then
            next:
        end if:
        sys := sort([f, g]):
        if not Duplicated(sys, systems) then
            try
                s := timelimit(max_exec_time, expand(charset(sys, xs))):
            catch "time expired":
                next:
            catch:
                next:
            end try:
            if s = [] then
                next:
            end if:
            systems := [op(systems), sys]:
            break: 
        end if:
    end do:
    print(f);
    print("#");
    print(g);
    print("#");
    s_str := "[":
    for i to nops(s) do
        s_str := cat(s_str, "'", convert(s[i], string), "'"):
        if i <> nops(s) then
            s_str := cat(s_str, ","):
        end if:
    end do:
    s_str := cat(s_str, "]"):
    print(s_str);
    print("#");
    return:
end proc:

randomize(0):
with(ListTools):
interface(prettyprint = 0):
xs := [seq(cat('x', i), i=0..{v - 1})]:
for i to {N} do
    GenerateSample(xs, {t}, {e}, {F});
end do:
    '''
    mpl_file_path = join(abspath(dirname(__file__)), 'v' + str(v) +
                         't' + str(t) + 'e' + str(e) + 'F' + str(F) + '.mpl')
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
parser.add_argument('--e', type=int, default=2,
                    help='the maximum exponent number in input polynomials (default: 2)')
parser.add_argument('--F', type=int, default=7,
                    help='the modulus in a finite field (default: 7)')
args = parser.parse_args()

# Dataset folder
base_dir = join(abspath(dirname(__file__)), '..', '..', '..', 'dataset', 'charset',
                'v' + str(args.v) + 't' + str(args.t) + 'e' + str(args.e) + 'F' + str(args.F))
generate_folder(base_dir)

# Generate dataset
data = run_maple_with_param(args.N, args.v, args.t, args.e, args.F)
for i in range(0, args.N * 3, 3):
    sample = CharSetData([data[i], data[i + 1]], eval(data[i + 2]))
    save(sample, join(base_dir, str(i // 3) + '.pt'))
