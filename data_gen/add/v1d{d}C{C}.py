'''
    Generate datasets of multiplication of two univariate polynomials\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 29 Dec 2025
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
from data_utils.storage import AddData  # noqa
from torch import save


def run_maple_with_param(N: int, d: int, C: int) -> List:
    r'''Write a Maple script and run it.

    Returns a list of samples.

    Args:
        N (int): Dataset size.
        d (int): 2 ^ d - 1 denotes degree of a polynomial.
        C (int): The maximum number of digit(s) in a coefficient.
    '''
    deg = 2 ** d - 1
    maple_script = f'''
LegalCheck := proc(p::polynom, x::name)
    if p = 0 then
        return false:
    end if:
    return true:
end proc:

ObtainTerms := proc(p::polynom)
    if not type(p, `+`) then
        return [p]:
    end if:
    return [op(p)]:
end proc:

# Generate a random univariate polynomial
RandPoly := proc(x::name, deg::integer)
    local p:
    while true do
        p := randpoly(x, coeffs=rand(-{10 ** C - 1}..{10 ** C - 1}), dense, degree=deg):
        if LegalCheck(p, x) then
            break:
        end if:
    end do:
    return p:
end proc:

# Obtain a table of coefficients of like terms in collected sum, indexed by exponent numbers of the variable
AddTable := proc(p1::polynom, p2::polynom, x::name)
    local T, p1t, p2t, t, exp, c:
    T := table():
    p1t := ObtainTerms(p1):
    for t in p1t do
        exp := degree(t, x):
        c := coeff(t, x, exp):
        if not assigned(T[exp]) then
            T[exp] := []:
        end if:
        T[exp] := [op(T[exp]), c]:
    end do:
    p2t := ObtainTerms(p2):
    for t in p2t do
        exp := degree(t, x):
        c := coeff(t, x, exp):
        if not assigned(T[exp]) then
            T[exp] := []:
        end if:
        T[exp] := [op(T[exp]), c]:
    end do:
    return T:
end proc:

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
                if i > 0 then
                    term := ""||term||"*x^"||i:
                end if:
            else
                coeffs := "":
                for j from 1 to nops(like_term) do
                    coeffs := cat(
                        coeffs,
                        if like_term[j] >= 0 then "+" else "" fi,
                        convert(like_term[j], string)
                    ):
                end do:
                if i > 0 then
                    term := "("||coeffs||")*x^"||i:
                else
                    term := "("||coeffs||")":
                end if:
            end if:
            terms := [op(terms), term]:
        end if:
    end do:
    return cat(op(terms)):
end proc:

CollectedSum := proc(p1::polynom, p2::polynom, x::name)
    return Format(AddTable(p1, p2, x), x)
end proc:

# Whether a system is duplicated
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
GenerateSample := proc(deg::integer)
    local p1, p2, sys:
    global systems:
    while true do
        p1 := RandPoly(x, deg):
        p2 := RandPoly(x, deg):
        sys := sort([p1, p2]):
        if not Duplicated(sys, systems) then
            systems := [op(systems), sys]:
            print(p1);
            print("#");
            print(p2);
            print("#");
            print(CollectedSum(p1, p2, x));
            print("#");
            print(expand(p1 + p2));
            print("#");
            break:
        end if:
    end do:
    return:
end proc:

randomize(0):
interface(prettyprint = 0):
for i from 1 to {N} do
    GenerateSample({deg});
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
parser.add_argument('--d', type=int, default=2,
                    help='2 ^ d - 1 denotes degree of a polynomial (default: 2)')
parser.add_argument('--C', type=int, default=2,
                    help='the maximum number of digit(s) in a coefficient (default: 1)')
args = parser.parse_args()
base_dir = join(abspath(dirname(__file__)), '..', '..', '..',
                'dataset', 'add', 'v1d' + str(args.d) + 'C' + str(args.C))
generate_folder(base_dir)

data = run_maple_with_param(args.N, args.d, args.C)
for i in range(0, args.N * 4, 4):
    sample = AddData([data[i], data[i + 1]], data[i + 2], data[i + 3])
    save(sample, join(base_dir, str(i // 4) + '.pt'))
