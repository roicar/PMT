'''
    Generate datasets of multiplication of two dense multivariate polynomials
    Coded and annotated by Yuxuan SONG
    Last modified time: 14 Jun 2026
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
from data_utils.storage import MulData  # noqa
from torch import save


def run_maple_with_param(N: int, v: int, d: int, C:int) -> List:
    r'''Write a Maple script and run it.

    Returns a list of samples.

    Args:
        N (int): Dataset size.
        v (int): The number of variables in a polynomial.
        d (int): The degree of a polynomial.
        C (int): The maximum number of digit(s) in a coefficient.
    '''
    maple_script = f'''
LegalCheck := proc(f::polynom, xs::list, deg::integer)
    if degree(f, xs) = deg then
        return true:
    end if:
    return false:
end proc:

RandPoly := proc(xs::list, deg::integer)
    local f:
    while true do
        f := randpoly(xs, coeffs=rand(-{10 ** C - 1}..{10 ** C - 1}), dense, degree=deg):
        if LegalCheck(f, xs, deg) then
            break:
        end if:
    end do:
    return f:
end proc:

# Whether a system is duplicated
Duplicated := proc(sys::list, L::list)
    local i:
    for i to nops(L) do
        if sys = L[i] then
            return true:
        end if:
    end do:
    return false:
end proc:

ObtainTerms := proc(f::polynom)
    if not type(f, `+`) then
        return [f]:
    end if:
    return [op(f)]:
end proc:

# Obtain a table of coefficients of like terms in collected product
ProductTable := proc(f::polynom, g::polynom, xs::list)
    local T, ft, pt, c, m:
    T := table():
    for ft in ObtainTerms(f) do
        for pt in ObtainTerms(expand(ft * g)) do
            c := coeffs(pt, xs, 'monomials'):
            m := pt / c:
            if not assigned(T[m]) then
                T[m] := []:
            end if:
            T[m] := [op(T[m]), c]:
        end do:
    end do:
    return T:
end proc:

grlex_desc := proc(a, b, xs::list)
    local deg_a, deg_b, i, deg_a_i, deg_b_i:
    
    deg_a := degree(a):
    deg_b := degree(b):
    if deg_a > deg_b then return true:
    elif deg_a < deg_b then return false:
    end if:
    
    for i in xs do
        deg_a_i := degree(a, i):
        deg_b_i := degree(b, i):
        if deg_a_i > deg_b_i then return true:
        elif deg_a_i < deg_b_i then return false:
        end if:
    end do:
    
    return false:
end proc:

Format := proc(T::table, xs::list)
    local ms, terms, m, like_term, coeffs, term, j:
    ms := map(op, [indices(T)]):
    ms := sort(ms, (a,b) -> grlex_desc(a, b, xs)):
    terms := []:
    for m in ms do
        like_term := T[m]:
        
        if nops(like_term) = 1 then
            if like_term[1] >= 0 then
                coeffs := sprintf("+%d", like_term[1]):
            else
                coeffs := sprintf("%d", like_term[1]):
            end if:
            
            if m = 1 then
                term := coeffs:
            else
                term := sprintf("%s*%s", coeffs, convert(m, string)):
            end if:
        else
            coeffs := "":
            for j to nops(like_term) do
                if like_term[j] >= 0 then
                    coeffs := cat(coeffs, sprintf("+%d", like_term[j])):
                else
                    coeffs := cat(coeffs, sprintf("%d", like_term[j])):
                end if:
            end do: 
            coeffs := convert(coeffs, string):
            term := sprintf("(%s)*%s", coeffs, convert(m, string)):
        end if: 
        terms := [op(terms), term]:
    end do:
    return convert(cat(op(terms)), string):
end proc:

CollectedProduct := proc(f::polynom, g::polynom, xs::list)
    return Format(ProductTable(f, g, xs), xs)
end proc:

systems := []:
GenerateSample := proc(xs::list, deg::integer)
    local f, g, sys:
    global systems:
    while true do
        f := RandPoly(xs, deg):
        g := RandPoly(xs, deg):
        sys := sort([f, g]):
        if not Duplicated(sys, systems) then
            systems := [op(systems), sys]:
            break:
        end if:
    end do:
    print(f);
    print("#");
    print(g);
    print("#");
    print(expand(f * g));
    print("#");
    print(CollectedProduct(f, g, xs));
    print("#");
    return:
end proc:

randomize(0):
interface(prettyprint = 0):
xs := [seq(cat('x', i), i=0..{v - 1})]:
for i to {N} do
    GenerateSample(xs, {d});
end do:
    '''
    mpl_file_path = join(abspath(dirname(__file__)), 'v' + str(v) +
                         'd' + str(d) + 'C' + str(C) + '.mpl')
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
parser.add_argument('--d', type=int, default=2,
                    help='the degree of a polynomial (default: 2)')
parser.add_argument('--C', type=int, default=1,
                    help='the maximum number of digit(s) in a coefficient (default: 1)')
args = parser.parse_args()

# Dataset folder
base_dir = join(abspath(dirname(__file__)), '..', '..', '..',
                'dataset', 'mul', 'v' + str(args.v) + 'd' + str(args.d) + 'C' + str(args.C))
generate_folder(base_dir)

# Generate dataset
data = run_maple_with_param(args.N, args.v, args.d, args.C)
for i in range(0, args.N * 4, 4):
    sample = MulData([data[i], data[i + 1]], data[i + 2], data[i + 3])
    save(sample, join(base_dir, str(i // 4) + '.pt'))
