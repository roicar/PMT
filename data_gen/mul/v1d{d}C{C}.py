'''
    Generate datasets of multiplication of two univariate polynomials\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 20 Feb 2026
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


def run_maple_with_param(N: int, d: int, C: int) -> List:
    r'''Write a Maple script and run it.

    Returns a list of samples.

    Args:
        N (int): Dataset size.
        d (int): The degree of a polynomial.
        C (int): The maximum number of digit(s) in a coefficient.
    '''
    maple_script = f'''
LegalCheck := proc(f::polynom, x::name, deg::integer)
    if degree(f, x) = deg then
        return true:
    end if:
    return false:
end proc:

# Generate a random univariate polynomial
RandPoly := proc(x::name, deg::integer)
    local f:
    while true do
        f := randpoly(x, coeffs=rand(-{10 ** C - 1}..{10 ** C - 1}), dense, degree=deg):
        if LegalCheck(f, x, deg) then
            break:
        end if:
    end do:
    return f:
end proc:

ObtainTerms := proc(f::polynom)
    if not type(f, `+`) then
        return [f]:
    end if:
    return [op(f)]:
end proc:

# Obtain a table of coefficients of like terms in collected product
ProductTable := proc(f::polynom, g::polynom, x::name)
    local T, ft, pt, exp, c:
    T := table():
    for ft in ObtainTerms(f) do
        for pt in ObtainTerms(expand(ft * g)) do
            exp := degree(pt, x):
            c := coeff(pt, x, exp):
            if not assigned(T[exp]) then
                T[exp] := []:
            end if:
            T[exp] := [op(T[exp]), c]:
        end do:
    end do:
    return T:
end proc:

# Example:
# (5*x^3+3*x^2-4*x-8) * (2*x^3-6*x^2-4*x-5) = 
# = 10*x^6-30*x^5-20*x^4-25*x^3+6*x^5-18*x^4-12*x^3-15*x^2-8*x^4+24*x^3+16*x^2+20*x-16*x^3+48*x^2+32*x+40
# = +10*x^6(-30+6)*x^5(-20-18-8)*x^4(-25-12+24-16)*x^3(-15+16+48)*x^2(+20+32)*x^1+40
Format := proc(T::table, x::name)
    local terms, deg, i, like_term, coeffs, term, j:
    terms := []:
    deg := max(indices(T)):
    for i from deg by -1 to 0 do
        if assigned(T[i]) then
            like_term := T[i]:
            if nops(like_term) = 1 then
                # Explicit positive sign
                coeffs := cat(
                    if like_term[1] >= 0 then "+" else "" fi,
                    convert(like_term[1], string)
                ):
                term := coeffs:
                if i >= 1 then
                    term := ""||term||"*x^"||i:
                end if:
            else
                coeffs := "":
                for j to nops(like_term) do
                    coeffs := cat(
                        coeffs,
                        if like_term[j] >= 0 then "+" else "" fi,
                        convert(like_term[j], string)
                    ):
                end do:
                term := "("||coeffs||")*x^"||i:
            end if:
            if nops(terms) = 1 then
                terms := [terms[1], term]:
            else:
                terms := [op(terms), term]:
            end if:
        end if:
    end do:
    if nops(terms) = 1 then
        return terms
    end if:
    return cat(op(terms)):
end proc:

CollectedProduct := proc(f::polynom, g::polynom, x::name)
    return Format(ProductTable(f, g, x), x)
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

systems := []:
GenerateSample := proc(deg::integer)
    local f, g, x, sys, prod, collected_prod:
    global systems:
    while true do
        f := RandPoly(x, deg):
        g := RandPoly(x, deg):
        sys := sort([f, g]):
        if not Duplicated(sys, systems) then
            systems := [op(systems), sys]:
            print(f);
            print("#");
            print(g);
            print("#");
            prod := expand(f * g):
            print(prod);
            print("#");
            if deg > 1 then
                collected_prod := CollectedProduct(f, g, x):
                print(collected_prod);
                print("#");
            end if:
            break:
        end if:
    end do:
    return:
end proc:

randomize(0):
interface(prettyprint = 0):
for i to {N} do
    GenerateSample({d});
end do:
    '''
    mpl_file_path = join(abspath(dirname(__file__)),
                         'v1d' + str(d) + 'C' + str(C) + '.mpl')
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
parser.add_argument('--d', type=int, default=1,
                    help='the degree of a polynomial (default: 1)')
parser.add_argument('--C', type=int, default=1,
                    help='the maximum number of digit(s) in a coefficient (default: 1)')
args = parser.parse_args()
base_dir = join(abspath(dirname(__file__)), '..', '..', '..',
                'dataset', 'mul', 'v1d' + str(args.d) + 'C' + str(args.C))
generate_folder(base_dir)

data = run_maple_with_param(args.N, args.d, args.C)
if args.d == 1:
    for i in range(0, args.N * 3, 3):
        sample = MulData([data[i], data[i + 1]], data[i + 2])
        save(sample, join(base_dir, str(i // 3) + '.pt'))
else:
    for i in range(0, args.N * 4, 4):
        sample = MulData([data[i], data[i + 1]], data[i + 2], data[i + 3])
        save(sample, join(base_dir, str(i // 4) + '.pt'))
